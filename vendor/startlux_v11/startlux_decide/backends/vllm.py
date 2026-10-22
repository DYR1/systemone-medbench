"""Client of one or more vLLM OpenAI-compatible servers (``/v1/completions``), the default engine.

Read-out: one prompt, ``max_tokens`` 1, ``logprobs`` k with ``allowed_token_ids`` = the k option letters and
``return_tokens_as_token_ids``. With ``--logprobs-mode processed_logprobs`` the returned top-k are exactly the k letters
(the mask is applied before the log-probabilities are taken), and ``--generation-config vllm`` keeps the folder's
generation_config (top_k, top_p) away from the read-out. ``start()`` checks both with a probe read-out, and reads every
server's vLLM version from ``/version``.

Thoughts: sampled with the AutoThink settings and stop string "</think>". On vLLM 0.31 or later a thought is one
streamed request (``generate_stream``; closing the stream early ends the request in vLLM). On vLLM 0.19.1 the state of
hybrid models (Gated DeltaNet layers, as in these models) gets corrupted when new generation requests join a decode
batch that is already running: 25 to 35% of thoughts turned into garbage loops in our tests. There thoughts are
generated in chunks, and the generation requests to one server are sent in waves (WaveScheduler): everything queued
goes out together, and the next wave starts only when the whole wave has finished. With waves no thought was corrupted
(0 of 183). Read-outs are prefill only and are sent at any time. vLLM 0.31 fixed the corruption: waves are on by
default only when a server runs an older vLLM or does not tell its version (``waves`` True or False decides instead).

Images (vLLM 0.31 or later, servers started with images allowed): a call whose prompt holds images goes to
``/v1/chat/completions`` with one user message whose parts are the pieces of the rendered prompt and the images (data
URIs), and with a chat template that writes those parts back to back and the image placeholder for each image
(images.PASS_THROUGH_TEMPLATE; the servers need ``--trust-request-chat-template``). vLLM then reads exactly the
package's prompt, with each image's tokens and positions in place. The resize bounds are the server's
(``--mm-processor-kwargs``, set by the launcher to the package's bounds; vLLM 0.31 refuses them per request).

Prefix cache (vLLM 0.31 or later, causal folders): ``prefill(prompt)`` reads a prompt into the cache (one generated
token, discarded), so that the questions of a request that share a long state read it once.

Every call of one question goes to the same server (``lane()``, round robin over questions).
"""
from __future__ import annotations

import asyncio
import itertools
import re
from typing import Any, AsyncIterator, Awaitable, Callable, Sequence

from ..images import PASS_THROUGH_TEMPLATE, Media, split_prompt
from ..prompt import check_tokenizer, from_systemone, messages, plain_prompt
from .base import Backend, BackendError, BackendRequestError, Generation, Sampling, StreamEvent
from .http import base_url, new_session, open_stream, request_json, sse_events

Transport = Callable[[str, Any], Awaitable[Any]]      # (url, JSON body or None for GET) -> JSON response; for a body
                                                      # with "stream": an async iterator of the JSON events
STREAM_VERSION = (0, 31)                              # vLLM from which thoughts are streamed and waves are not needed
CHAT = "/v1/chat/completions"                         # calls with images
COMPLETIONS = "/v1/completions"                       # everything else


def endpoint(body: dict) -> str:
    return CHAT if "messages" in body else COMPLETIONS


def parse_version(text: Any) -> tuple | None:
    """'0.31.0', '0.31.0rc1', '0.19.1+cu129' -> (0, 31, 0), (0, 31, 0), (0, 19, 1); None when there is no version."""
    m = re.match(r"\s*v?(\d+)\.(\d+)(?:\.(\d+))?", str(text or ""))
    return tuple(int(x) for x in m.groups(default="0")) if m else None


class WaveScheduler:
    """Sends the generation requests of one server in waves (see the module docstring): the worker takes the first
    queued request, waits `gather_s` seconds for more, sends up to `max_wave` together and waits for all of them before
    the next wave."""

    def __init__(self, send: Callable[[dict], Awaitable[dict]], max_wave: int = 128, gather_s: float = 0.3):
        self._send, self.max_wave, self.gather_s = send, max_wave, gather_s
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None

    async def submit(self, body: dict) -> dict:
        if self._task is None:
            self._queue = asyncio.Queue()
            self._task = asyncio.ensure_future(self._worker())
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put((body, fut))
        return await fut

    async def _worker(self) -> None:
        while True:
            batch = [await self._queue.get()]
            await asyncio.sleep(self.gather_s)
            while len(batch) < self.max_wave and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            batch = [(b, f) for b, f in batch if not f.done()]          # callers that gave up meanwhile
            results = await asyncio.gather(*(self._send(b) for b, _ in batch), return_exceptions=True)
            for (_, fut), r in zip(batch, results):
                if fut.done():
                    continue
                if isinstance(r, BaseException):
                    fut.set_exception(r)
                else:
                    fut.set_result(r)

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None


class _Lane(Backend):
    """The calls of one question, pinned to one server."""

    def __init__(self, backend: "VLLMBackend", url: str):
        self.backend, self.url = backend, url
        self.name, self.tokenizer = backend.name, backend.tokenizer

    async def readout(self, prompt: str, letter_ids: Sequence[int], media: Media | None = None) -> list[float]:
        return await self.backend.readout_at(self.url, prompt, letter_ids, media)

    async def generate(self, prompt: str, max_tokens: int, sampling: Sampling, stop: Sequence[str],
                       seed: int, media: Media | None = None) -> Generation:
        return await self.backend.generate_at(self.url, prompt, max_tokens, sampling, stop, seed, media)

    @property
    def streams(self) -> bool:
        return self.backend.streams

    @property
    def reads_images(self) -> bool:
        return self.backend.reads_images

    @property
    def caches_prefixes(self) -> bool:
        return self.backend.caches_prefixes

    def generate_stream(self, prompt: str, max_tokens: int, sampling: Sampling, stop: Sequence[str],
                        seed: int, media: Media | None = None) -> AsyncIterator[StreamEvent]:
        return self.backend.stream_at(self.url, prompt, max_tokens, sampling, stop, seed, media)

    async def prefill(self, prompt: str, media: Media | None = None) -> None:
        await self.backend.prefill_at(self.url, prompt, media)


class VLLMBackend(Backend):
    """See the module docstring. `urls`: server base URLs (or bare ports on 127.0.0.1); `model_name`: the served model
    name (default: what the servers report); `waves`: None decides from the servers' vLLM version; `images`: send
    images to servers that take them (vLLM 0.31 or later); `transport` replaces HTTP (tests)."""

    name = "vllm"

    def __init__(self, urls: Sequence[str], tokenizer: Any, model_name: str | None = None, waves: bool | None = None,
                 wave_max: int = 128, wave_gather: float = 0.3, timeout: float = 3600.0, attempts: int = 4,
                 retry_delay: float = 1.0, transport: Transport | None = None, images: bool = True):
        if not urls:
            raise ValueError("at least one vLLM server URL is needed")
        self.urls = [base_url(u) for u in urls]
        self.tokenizer = tokenizer
        self.model_name = model_name
        self._auto_waves = waves is None
        self.waves, self.wave_max, self.wave_gather = waves is not False, wave_max, wave_gather
        self.versions: dict[str, str | None] = {}
        self.streams = self.reads_images = self.caches_prefixes = False
        self._images = images
        self.timeout, self.attempts, self.retry_delay = timeout, attempts, retry_delay
        self._transport = transport
        self._session = None
        self._next = itertools.cycle(self.urls)
        self._schedulers: dict[str, WaveScheduler] = {}

    # ---------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        """Open the HTTP session, find the served model name, context length and vLLM versions, and probe the
        read-out flags. Thoughts are streamed when every server runs vLLM 0.31 or later."""
        if self._transport is None and self._session is None:
            self._session = new_session(self.timeout)
        names, lengths = set(), set()
        for url in self.urls:
            data = (await self._request(url + "/v1/models", None)).get("data") or []
            if not data:
                raise BackendError(f"{url} serves no model")
            names.add(data[0].get("id"))
            lengths.add(data[0].get("max_model_len"))
            self.versions[url] = await self._version(url)
        found = [parse_version(v) for v in self.versions.values()]
        self.streams = all(v is not None and v >= STREAM_VERSION for v in found)
        self.reads_images = self._images and self.streams
        self.caches_prefixes = self.streams
        if self._auto_waves:
            self.waves = not self.streams
        if self.model_name is None:
            if len(names) != 1:
                raise BackendError(f"the vLLM servers serve different models {sorted(map(str, names))}")
            self.model_name = names.pop()
        lengths.discard(None)
        if len(lengths) > 1:
            raise BackendError(f"the vLLM servers have different context lengths {sorted(lengths)}")
        self.context_length = lengths.pop() if lengths else None
        await self._probe()

    async def _version(self, url: str) -> str | None:
        """The server's vLLM version from GET /version (one try; None when it does not answer with one)."""
        try:
            if self._transport is not None:
                out = await self._transport(url + "/version", None)
            else:
                async with self._session.get(url + "/version") as r:
                    if r.status != 200:
                        return None
                    out = await r.json(content_type=None)
        except Exception:                      # noqa: BLE001  (an old or unusual server: the version stays unknown)
            return None
        version = out.get("version") if isinstance(out, dict) else None
        return str(version) if version else None

    async def _probe(self) -> None:
        """A read-out over all 26 letters must return all 26 with finite log-probabilities."""
        letters = check_tokenizer(self.tokenizer)
        question = {"type": "noul", "instructions": "Is this a probe?"}
        probe = plain_prompt(self.tokenizer, messages(from_systemone("probe", question)))
        hint = ("start vLLM with --logprobs-mode processed_logprobs --max-logprobs 26 --generation-config vllm "
                "(startlux-decide launch-vllm does)")
        for url in self.urls:
            try:
                z = await self.readout_at(url, probe, letters)
            except BackendRequestError as e:
                raise BackendError(f"{url} refused a read-out over 26 letters ({e}); {hint}") from None
            if not all(v > -9999.0 for v in z):
                raise BackendError(f"{url} did not return all 26 option letters in a read-out; {hint}")

    async def close(self) -> None:
        for s in self._schedulers.values():
            await s.close()
        self._schedulers.clear()
        if self._session is not None:
            await self._session.close()
            self._session = None

    def lane(self) -> Backend:
        return _Lane(self, next(self._next))

    def describe(self) -> dict:
        return {"name": self.name, "servers": self.urls, "served_model": self.model_name,
                "vllm_versions": [self.versions.get(u) for u in self.urls], "streams": self.streams,
                "images": self.reads_images, "prefix_cache": self.caches_prefixes, "waves": self.waves,
                "context_length": self.context_length}

    # ---------------------------------------------------------------- primitives
    async def readout(self, prompt: str, letter_ids: Sequence[int], media: Media | None = None) -> list[float]:
        return await self.readout_at(next(self._next), prompt, letter_ids, media)

    async def generate(self, prompt: str, max_tokens: int, sampling: Sampling, stop: Sequence[str],
                       seed: int, media: Media | None = None) -> Generation:
        return await self.generate_at(next(self._next), prompt, max_tokens, sampling, stop, seed, media)

    def generate_stream(self, prompt: str, max_tokens: int, sampling: Sampling, stop: Sequence[str],
                        seed: int, media: Media | None = None) -> AsyncIterator[StreamEvent]:
        return self.stream_at(next(self._next), prompt, max_tokens, sampling, stop, seed, media)

    async def prefill(self, prompt: str, media: Media | None = None) -> None:
        await self.prefill_at(next(self._next), prompt, media)

    def _body(self, prompt: str, media: Media | None, **params: Any) -> dict:
        """A completions body, or with images a chat body whose user message holds the prompt's pieces and the
        images in order (module docstring)."""
        if media is None:
            return {"model": self.model_name, "prompt": prompt, **params}
        pieces = split_prompt(prompt)
        if len(pieces) != len(media) + 1:
            raise BackendRequestError(f"the prompt has {len(pieces) - 1} image places for {len(media)} images")
        parts: list = []
        for k, piece in enumerate(pieces):
            if piece:
                parts.append({"type": "text", "text": piece})
            if k < len(media):
                parts.append({"type": "image_url", "image_url": {"url": media.data_uris()[k]}})
        return {"model": self.model_name, "messages": [{"role": "user", "content": parts}],
                "chat_template": PASS_THROUGH_TEMPLATE, "add_generation_prompt": False, **params}

    async def readout_at(self, url: str, prompt: str, letter_ids: Sequence[int],
                         media: Media | None = None) -> list[float]:
        k = len(letter_ids)
        if media is None:
            body = self._body(prompt, None, max_tokens=1, temperature=1.0, logprobs=k,
                              allowed_token_ids=list(letter_ids), return_tokens_as_token_ids=True)
            out = await self._request(url + COMPLETIONS, body)
            top = out["choices"][0]["logprobs"]["top_logprobs"][0]
        else:
            body = self._body(prompt, media, max_tokens=1, temperature=1.0, logprobs=True, top_logprobs=k,
                              allowed_token_ids=list(letter_ids), return_tokens_as_token_ids=True)
            out = await self._request(url + CHAT, body)
            top = {e["token"]: e["logprob"] for e in out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
        return [top.get(f"token_id:{t}", -1e9) for t in letter_ids]

    def _generation_body(self, prompt: str, max_tokens: int, sampling: Sampling, stop: Sequence[str],
                         seed: int, media: Media | None = None) -> dict:
        return self._body(prompt, media, max_tokens=max_tokens, temperature=sampling.temperature,
                          top_p=sampling.top_p, top_k=sampling.top_k, stop=list(stop), seed=seed)

    async def generate_at(self, url: str, prompt: str, max_tokens: int, sampling: Sampling, stop: Sequence[str],
                          seed: int, media: Media | None = None) -> Generation:
        body = self._generation_body(prompt, max_tokens, sampling, stop, seed, media)
        if self.waves:
            if url not in self._schedulers:
                self._schedulers[url] = WaveScheduler(lambda b, u=url: self._request(u + endpoint(b), b),
                                                      self.wave_max, self.wave_gather)
            out = await self._schedulers[url].submit(body)
        else:
            out = await self._request(url + endpoint(body), body)
        c = out["choices"][0]
        text = c["text"] if "text" in c else ((c.get("message") or {}).get("content") or "")
        return Generation(text, (out.get("usage") or {}).get("completion_tokens", 0),
                          "stop" if c.get("finish_reason") == "stop" else "length")

    async def prefill_at(self, url: str, prompt: str, media: Media | None = None) -> None:
        """Read the prompt into the server's prefix cache: one greedy token, discarded."""
        body = self._body(prompt, media, max_tokens=1, temperature=0.0)
        await self._request(url + endpoint(body), body)

    async def stream_at(self, url: str, prompt: str, max_tokens: int, sampling: Sampling, stop: Sequence[str],
                        seed: int, media: Media | None = None) -> AsyncIterator[StreamEvent]:
        """One streamed generation request (the body of generate_at plus "stream"). Token counts come from vLLM's
        continuous usage statistics. Closing the iterator before its last event closes the connection, and vLLM
        aborts the request."""
        body = self._generation_body(prompt, max_tokens, sampling, stop, seed, media)
        body.update({"stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True}})
        response = None
        if self._transport is not None:
            events = await self._transport(url + endpoint(body), body)
        else:
            response = await open_stream(self._session, url + endpoint(body), body, self.attempts,
                                         self.retry_delay)
            events = sse_events(response)
        tokens, seen, finish, last, done = 0, 0, None, "", False
        try:
            try:
                async for ev in events:
                    usage = ev.get("usage")
                    choices = ev.get("choices") or []
                    if choices:
                        seen += 1
                    tokens = usage.get("completion_tokens", tokens) if usage else (seen if choices else tokens)
                    if not choices:
                        continue
                    text = choices[0].get("text") or (choices[0].get("delta") or {}).get("content") or ""
                    if choices[0].get("finish_reason") is not None:
                        finish, last = choices[0]["finish_reason"], last + text    # the usage event may follow
                    elif finish is None and text:
                        yield StreamEvent(text, tokens)
            except (BackendError, BackendRequestError):
                raise
            except Exception as e:                  # noqa: BLE001  (the connection broke or an event was bad)
                raise BackendError(f"{url}: the thought stream failed ({type(e).__name__}: {e})") from None
            if finish is None:
                raise BackendError(f"{url}: the thought stream ended without a finish reason")
            done = True
            yield StreamEvent(last, tokens, "stop" if finish == "stop" else "length")
        finally:
            if response is not None:
                if done:
                    response.release()
                else:
                    response.close()                # vLLM aborts a request whose client went away
            elif not done and hasattr(events, "aclose"):
                await events.aclose()

    # ---------------------------------------------------------------- HTTP
    async def _request(self, url: str, body: dict | None) -> dict:
        if self._transport is not None:
            return await self._transport(url, body)
        return await request_json(self._session, url, body, self.attempts, self.retry_delay)
