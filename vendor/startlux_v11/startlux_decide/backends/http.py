"""JSON over HTTP for the server-based engines (vLLM, llama-server)."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator

from .base import BackendError, BackendRequestError


def base_url(url: str) -> str:
    """'8001', 'host:8001' or 'http://host:8001[/v1][/]' -> 'http://host:8001'."""
    url = url.strip()
    if re.fullmatch(r"\d+", url):
        url = f"127.0.0.1:{url}"
    if "://" not in url:
        url = "http://" + url
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def new_session(timeout: float, connections: int = 2048) -> Any:
    import aiohttp
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout),
                                 connector=aiohttp.TCPConnector(limit=connections))


async def request_json(session: Any, url: str, body: dict | None, attempts: int, retry_delay: float = 1.0) -> dict:
    """POST `body` as JSON (GET when None) and return the JSON answer. A 400, 413 or 422 answer is final
    (BackendRequestError: the engine refused the request); connection errors, timeouts and other statuses are tried
    again after 1, 2, 3 ... times `retry_delay` seconds, up to `attempts` tries in all (BackendError after the last)."""
    import aiohttp
    last = "no attempt"
    for attempt in range(attempts):
        try:
            call = session.get(url) if body is None else session.post(url, json=body)
            async with call as r:
                if r.status == 200:
                    return await r.json()
                text = await r.text()
                if r.status in (400, 413, 422):
                    raise BackendRequestError(text[:300])
                last = f"HTTP {r.status}: {text[:200]}"
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last = f"{type(e).__name__}: {e}"
        if attempt + 1 < attempts:
            await asyncio.sleep(retry_delay * (1 + attempt))
    raise BackendError(f"{url}: no answer after {attempts} attempts ({last})")


async def open_stream(session: Any, url: str, body: dict, attempts: int, retry_delay: float = 1.0) -> Any:
    """POST `body` for a streamed answer and return the response once its status is 200; the caller reads it and
    closes it. Failures before that are handled as in request_json: 400, 413 and 422 are final (BackendRequestError),
    connection errors and other statuses are tried again, BackendError after the last try."""
    import aiohttp
    last = "no attempt"
    for attempt in range(attempts):
        try:
            r = await session.post(url, json=body)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last = f"{type(e).__name__}: {e}"
        else:
            if r.status == 200:
                return r
            text = await r.text()
            r.release()
            if r.status in (400, 413, 422):
                raise BackendRequestError(text[:300])
            last = f"HTTP {r.status}: {text[:200]}"
        if attempt + 1 < attempts:
            await asyncio.sleep(retry_delay * (1 + attempt))
    raise BackendError(f"{url}: no answer after {attempts} attempts ({last})")


async def sse_events(response: Any) -> AsyncIterator[dict]:
    """The JSON events of a server-sent-events response ("data: {...}" lines) up to "data: [DONE]"."""
    async for raw in response.content:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)
