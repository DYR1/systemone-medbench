"""Latency of a running decision server, per request and per phase.

Every request of a JSONL file (one /v1/systemone body per line; "id" and "gold" fields are ignored) is sent with
return_details, `concurrency` at a time, after `warmup` untimed requests. Per request the client measures the time
from sending to the full answer (``client_ms``) and the server reports its own timings (``timing_ms``): wall time,
the sum of the first read-outs of its questions, the span from the first thought's start to the last thought's end,
and the sum of the read-outs after a thought. The summary has n, mean, median, p80 and p95 of every metric
(percentiles by linear interpolation between order statistics).
"""
from __future__ import annotations

import asyncio
import csv
import json
import time
from typing import Iterable, Sequence

ROW_FIELDS = ("index", "id", "status", "client_ms", "wall_ms", "first_pass_sum_ms", "thought_span_ms",
              "after_thought_sum_ms", "questions", "thought_questions", "thought_tokens", "error")
METRICS = ("client_ms", "wall_ms", "first_pass_sum_ms", "thought_span_ms", "after_thought_sum_ms", "thought_tokens")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def request_body(item: dict, controls: dict | None = None) -> dict:
    """The /v1/systemone body of a file item: everything but "id" and "gold", plus `controls`, with details on."""
    body = {k: v for k, v in item.items() if k not in ("id", "gold")}
    body.update(controls or {})
    body["return_details"] = True
    return body


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0..100), linear interpolation between order statistics."""
    v = sorted(values)
    if not v:
        return float("nan")
    pos = (len(v) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def summarize(rows: Iterable[dict], metrics: Sequence[str] = METRICS) -> list[dict]:
    """n, mean, median, p80 and p95 of every metric over the rows that succeeded."""
    ok = [r for r in rows if r["status"] == "ok"]
    out = []
    for m in metrics:
        v = [float(r[m]) for r in ok if r.get(m) not in (None, "")]
        out.append({"metric": m, "n": len(v), "mean": sum(v) / len(v) if v else float("nan"),
                    "median": percentile(v, 50), "p80": percentile(v, 80), "p95": percentile(v, 95)})
    return out


def row_of(index: int, item: dict, status: str, client_ms: float, response: dict | None, error: str = "") -> dict:
    row = dict.fromkeys(ROW_FIELDS, "")
    row.update({"index": index, "id": item.get("id", index), "status": status, "client_ms": round(client_ms, 3),
                "error": error})
    if response:
        t = response.get("timing_ms") or {}
        details = [a.get("details") or {} for a in (response.get("answers") or {}).values()]
        thoughts = [d["thought"] for d in details if d.get("thought", {}).get("used")]
        row.update({"wall_ms": t.get("wall", ""), "first_pass_sum_ms": t.get("first_pass_sum", ""),
                    "thought_span_ms": t.get("thought_span", ""),
                    "after_thought_sum_ms": t.get("after_thought_sum", ""),
                    "questions": len(details), "thought_questions": len(thoughts),
                    "thought_tokens": sum(th["tokens"] for th in thoughts)})
    return row


async def post_all(url: str, bodies: Sequence[dict], concurrency: int,
                   timeout: float) -> list[tuple[str, float, dict | None, str]]:
    """POST every body to url + /v1/systemone, `concurrency` at a time. -> [(status, client ms, response, error)]."""
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    out: list = [None] * len(bodies)

    async def one(i: int, session) -> None:
        async with sem:
            t0 = time.perf_counter()
            try:
                async with session.post(url.rstrip("/") + "/v1/systemone", json=bodies[i]) as r:
                    text = await r.text()
                    ms = (time.perf_counter() - t0) * 1000
                    if r.status == 200:
                        out[i] = ("ok", ms, json.loads(text), "")
                    else:
                        out[i] = (f"http_{r.status}", ms, None, text[:300])
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                out[i] = ("error", (time.perf_counter() - t0) * 1000, None, f"{type(e).__name__}: {e}")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout),
                                     connector=aiohttp.TCPConnector(limit=max(concurrency, 1) * 2)) as session:
        await asyncio.gather(*(one(i, session) for i in range(len(bodies))))
    return out


async def bench(url: str, items: Sequence[dict], concurrency: int = 1, warmup: int = 0, timeout: float = 3600.0,
                controls: dict | None = None) -> list[dict]:
    """Warm up, then time every item. -> one row per item (ROW_FIELDS)."""
    bodies = [request_body(it, controls) for it in items]
    if warmup:
        await post_all(url, [bodies[i % len(bodies)] for i in range(warmup)], concurrency, timeout)
    results = await post_all(url, bodies, concurrency, timeout)
    return [row_of(i, it, *res) for i, (it, res) in enumerate(zip(items, results))]


def write_csv(path: str, rows: Sequence[dict], fields: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in fields})
