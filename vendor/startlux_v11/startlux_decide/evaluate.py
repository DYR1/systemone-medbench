"""Accuracy and calibration of a decision server on gold-labelled requests, and the paired comparison of two runs
(for example BF16 against an FP8, GGUF or MLX build of the same model).

Input: JSONL, one request per line: {"id" (optional), "state", "questions", "gold": {question key: label}} where the
label is an option id for a choice, true/false (or "yes"/"no") for a yes/no question, and the level index (0 =
lowest) or a criteria key for a score. Questions without a gold label are sent but not scored.

Per question: predicted label = the choice / yes when P(yes) >= 0.5 / the most probable level; confidence = the
probability of the predicted label. Metrics: accuracy; ECE (top-label, 10 equal-width confidence bins, weighted by
bin size); Brier score (sum over the options of (p - 1[gold])^2, so a yes/no question counts both options); the share
of questions that thought and the mean thought tokens (from the answers' details).

Comparison (pairs of the same request id and question key with the same gold label): agreement of the predicted
labels, accuracy of each run, the difference B - A with a 95% bootstrap interval that resamples requests (questions of
one request share their evidence), agreement of the gate's decisions, and the largest change of any option's
probability per question (mean and max over questions).
"""
from __future__ import annotations

import json
import random
from typing import Any, Sequence

from .bench import percentile, post_all, request_body
from .prompt import score_keys

ECE_BINS = 10


def gold_label(spec: dict, gold: Any) -> str:
    """The gold label of a question in the form predicted labels take; raises ValueError."""
    t = spec.get("type", "choice")
    crit = spec.get("criteria", spec.get("options"))
    if t in ("noul", "bool"):
        mapping = {True: "true", False: "false", "true": "true", "false": "false", "yes": "true", "no": "false",
                   1: "true", 0: "false"}
        key = gold.lower() if isinstance(gold, str) else gold
        if key not in mapping:
            raise ValueError(f"a yes/no gold label must be true or false, not {gold!r}")
        return mapping[key]
    if t == "score":
        n = len(crit or [])
        if isinstance(gold, str) and isinstance(crit, dict) and gold in crit:
            return str(score_keys(crit).index(gold))
        if isinstance(gold, int) and not isinstance(gold, bool) and 0 <= gold < n:
            return str(gold)
        raise ValueError(f"a score gold label must be a level index 0..{n - 1} or a criteria key, not {gold!r}")
    options = [str(c) for c in crit] if isinstance(crit, (list, tuple, dict)) else []
    if str(gold) not in options:
        raise ValueError(f"the gold label {gold!r} is not an option of the question")
    return str(gold)


def prediction(question_type: str, answer: dict) -> tuple[str, dict]:
    """(predicted label, {label: probability}) of an answer."""
    if question_type in ("noul", "bool"):
        p = answer["noul"]
        return ("true" if p >= 0.5 else "false"), {"true": p, "false": 1 - p}
    probs = answer["probabilities"]
    if question_type == "score":
        return max(probs, key=probs.get), probs
    return answer["choice"], probs


def record(item: dict, index: int) -> dict:
    """A line of a predictions file before the answer: the request id, every question's type and gold label."""
    gold = item.get("gold") or {}
    questions = {}
    for key, spec in item["questions"].items():
        try:
            label = gold_label(spec, gold[key]) if gold.get(key) is not None else None
        except ValueError as e:
            raise ValueError(f"request {item.get('id', index)}, question {key}: {e}") from None
        questions[key] = {"type": spec.get("type", "choice"), "gold": label}
    return {"id": item.get("id", index), "status": "pending", "questions": questions}


async def run(url: str, items: Sequence[dict], concurrency: int = 8, timeout: float = 3600.0,
              controls: dict | None = None) -> list[dict]:
    """Send every item (with return_details) and return the prediction records. Every gold label is checked before
    anything is sent."""
    records = [record(it, i) for i, it in enumerate(items)]
    results = await post_all(url, [request_body(it, controls) for it in items], concurrency, timeout)
    for rec, (status, _, response, err) in zip(records, results):
        rec["status"] = status
        if response:
            rec["answers"], rec["timing_ms"] = response["answers"], response.get("timing_ms")
        if err:
            rec["error"] = err
    return records


def _rows(records: Sequence[dict]) -> list[dict]:
    """One row per scored question: request id, key, gold, predicted label, probabilities, thought."""
    rows = []
    for rec in records:
        for key, q in rec["questions"].items():
            if q["gold"] is None or rec["status"] != "ok":
                continue
            ans = rec["answers"][key]
            pred, probs = prediction(q["type"], ans)
            thought = (ans.get("details") or {}).get("thought") or {}
            rows.append({"id": rec["id"], "key": key, "type": q["type"], "gold": q["gold"], "pred": pred,
                         "probs": probs, "thought": thought.get("used"), "tokens": thought.get("tokens", 0)})
    return rows


def score(records: Sequence[dict]) -> dict:
    """Metrics of one predictions file (see the module docstring)."""
    rows = _rows(records)
    out = {"requests": len(records), "failed_requests": sum(r["status"] != "ok" for r in records),
           "questions": len(rows)}
    if not rows:
        return out
    bins = [[0, 0.0, 0.0] for _ in range(ECE_BINS)]           # count, sum of confidence, correct
    brier = 0.0
    for r in rows:
        conf, ok = r["probs"][r["pred"]], r["pred"] == r["gold"]
        b = bins[min(int(conf * ECE_BINS), ECE_BINS - 1)]
        b[0], b[1], b[2] = b[0] + 1, b[1] + conf, b[2] + ok
        brier += sum((p - (label == r["gold"])) ** 2 for label, p in r["probs"].items())
        brier += 0.0 if r["gold"] in r["probs"] else 1.0
    n = len(rows)
    out.update({"accuracy": 100 * sum(r["pred"] == r["gold"] for r in rows) / n,
                "ece": 100 * sum(abs(c - k) for _, c, k in bins) / n, "brier": brier / n})
    with_details = [r for r in rows if r["thought"] is not None]
    if with_details:
        thinking = [r for r in with_details if r["thought"]]
        out.update({"thought_share": 100 * len(thinking) / len(with_details),
                    "thought_tokens_mean": sum(r["tokens"] for r in with_details) / len(with_details),
                    "thought_tokens_mean_when_thinking": (sum(r["tokens"] for r in thinking) / len(thinking)
                                                          if thinking else 0.0)})
    by_type = {}
    for t in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == t]
        by_type[t] = {"questions": len(sub), "accuracy": 100 * sum(r["pred"] == r["gold"] for r in sub) / len(sub)}
    out["by_type"] = by_type
    return out


def compare(a: Sequence[dict], b: Sequence[dict], resamples: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of two predictions files of the same requests (see the module docstring)."""
    rows_a = {(r["id"], r["key"]): r for r in _rows(a)}
    pairs = [(ra, rb) for rb in _rows(b) for ra in [rows_a.get((rb["id"], rb["key"]))]
             if ra is not None and ra["gold"] == rb["gold"]]
    if not pairs:
        return {"paired_questions": 0}
    n = len(pairs)
    clusters: dict = {}
    for ra, rb in pairs:
        c = clusters.setdefault(ra["id"], [0, 0, 0])
        c[0] += ra["pred"] == ra["gold"]
        c[1] += rb["pred"] == rb["gold"]
        c[2] += 1
    acc_a = 100 * sum(c[0] for c in clusters.values()) / n
    acc_b = 100 * sum(c[1] for c in clusters.values()) / n
    rng, cl, diffs = random.Random(seed), list(clusters.values()), []
    for _ in range(resamples):
        draw = [cl[rng.randrange(len(cl))] for _ in cl]
        m = sum(c[2] for c in draw)
        diffs.append(100 * (sum(c[1] for c in draw) - sum(c[0] for c in draw)) / m)
    dp = [max(abs(ra["probs"].get(k, 0.0) - rb["probs"].get(k, 0.0)) for k in set(ra["probs"]) | set(rb["probs"]))
          for ra, rb in pairs]
    gated = [(ra, rb) for ra, rb in pairs if ra["thought"] is not None and rb["thought"] is not None]
    out = {"paired_questions": n, "paired_requests": len(clusters),
           "agreement": 100 * sum(ra["pred"] == rb["pred"] for ra, rb in pairs) / n,
           "accuracy_a": acc_a, "accuracy_b": acc_b, "accuracy_diff_b_minus_a": acc_b - acc_a,
           "accuracy_diff_ci95": [percentile(diffs, 2.5), percentile(diffs, 97.5)],
           "both_correct": sum(ra["pred"] == ra["gold"] and rb["pred"] == rb["gold"] for ra, rb in pairs),
           "only_a_correct": sum(ra["pred"] == ra["gold"] and rb["pred"] != rb["gold"] for ra, rb in pairs),
           "only_b_correct": sum(ra["pred"] != ra["gold"] and rb["pred"] == rb["gold"] for ra, rb in pairs),
           "mean_max_abs_dp": sum(dp) / n, "max_abs_dp": max(dp)}
    if gated:
        out["thought_agreement"] = 100 * sum(ra["thought"] == rb["thought"] for ra, rb in gated) / len(gated)
    return out


def write_jsonl(path: str, records: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
