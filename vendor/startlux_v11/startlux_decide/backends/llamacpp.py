"""Client of llama.cpp's llama-server (``/completion``) for GGUF builds of causal (AutoThink) folders.

The model folder still provides the tokenizer, config.json and decision_config.json; prompts are tokenized here and
sent as token ids.

Read-out: ``n_predict`` 1 with ``post_sampling_probs``, a logit bias of +100 on the k option letters and ``top_k`` = k,
so the sampler keeps exactly the k letters, and their returned probabilities are the softmax over the letters alone
(adding the same bias to every letter does not change their relative probabilities). Temperature 1 and every other
sampler neutral (top_p 1, min_p 0, no penalties). The probabilities are renormalised over the letters and returned
as log-probabilities.

Thoughts: ``/completion`` with the AutoThink sampling settings (min_p 0, as in vLLM), the stop string and the seed,
and the sampler order of vLLM: temperature first, then top-k and top-p on the tempered distribution (llama.cpp's
default chain applies top-p before the temperature, which keeps more tokens at temperature 0.6). Prompt caching is
off for every call (``cache_prompt`` false), as prefix caching is off in vLLM.

Instant (prompt-bidirectional) folders need non-causal attention over the prompt in the full-attention layers.
llama.cpp does this with a non-causal context; the prompt must then be evaluated alone and in one micro-batch.
``STARTLUX_LLAMACPP_BIDIR`` says how the server provides it:

* ``auto`` (default): ``start()`` reads the server's ``/props`` once. A server that reports
  ``mtmd_prompt_whole_when_non_causal`` has patch ``0003`` (below), which exists for non-causal contexts and is the only
  server that reads an Instant prompt with images whole: it works as ``server``. Every other server works as
  ``request``. llama.cpp's own ``/props`` says nothing about causal attention, so a stock llama-server is a ``request``
  server here and the first read-out fails with a message that names ``server``: set ``server`` for it. The chosen mode
  is in ``describe()`` and the log;
* ``request``: the server has the patch ``0001-server-bidirectional-prompts``; every read-out asks for it with
  ``"prompt_bidirectional": -1`` and the server's echo is checked (a stock server ignores the field: error);
* ``server``: the server's context is non-causal (a GGUF converted with ``attention.causal = false``, or
  ``--override-kv qwen35.attention.causal=bool:false``); a stock llama-server then needs ``--parallel 1``,
  ``--ctx-checkpoints 0`` and ``-ub`` at least the longest prompt;
* ``off``: read the prompt causally (a different model than the one trained; for measuring that cost only).

Images (llama-server started with ``--mmproj`` of the model; ``/props`` then reports the vision modality and the
server's media marker): a call whose prompt holds images sends ``"prompt": {"prompt_string": ..., "multimodal_data":
[...]}``: the rendered prompt as text with the server's media marker at each image's place, and each image as a
base64 PNG. llama.cpp adds ``<|vision_start|>`` and ``<|vision_end|>`` around every image itself and tokenizes the
text with its own tokenizer (an object prompt cannot carry token ids). The engine resizes every image with PIL
(bicubic) to the processor size of the request (images.Media.sizes: multiples of 32, at most ``max_pixels``) before it
sends it, so that the server's own resize does nothing and the pixels are the ones the other engines read; start
llama-server with ``--image-min-tokens`` and ``--image-max-tokens`` equal to those bounds in tokens (64 and 1024 for
65,536 and 1,048,576 pixels) so that it would not resize either. After every image call the engine compares the
server's ``tokens_evaluated`` with its own count (text tokens + 2 + the image's tokens per image) and raises when they
differ (a server that resized an image, or counts it differently); a different count of the text only (the two
tokenizers split a rare word differently) is told apart with the server's ``/tokenize``.

Instant folders and images: the image's embeddings must be read together with the text in ONE micro-batch with
non-causal attention, as in training. The server's context must be non-causal (the release GGUFs carry
``qwen35.attention.causal = false``, or ``--override-kv qwen35.attention.causal=bool:false``): use mode ``server``
with ``--parallel 1 --ctx-checkpoints 0 -b N -ub N``. The per-request mode (``request``, patch 0001) does not cover
images on a causal context: the engine refuses that combination. A STOCK llama-server (b11549) decodes the text before
an image, the image and the text after it as three separate micro-batches (only embedding tasks take its mixed-batch
path), which cuts the bidirectional attention: on 32 image questions of the 4B Instant model its read-outs differed
from the BF16 model's by 0.022 on average (largest 0.22), against 0.0055 (largest 0.06) with patch 0003
(``0003-server-mtmd-prompt-whole-on-non-causal-context``: a prompt with images on a non-causal context goes whole into
one batch, and ``/props`` says so in ``mtmd_prompt_whole_when_non_causal``). ``start()`` warns when an Instant folder
is served with images by a server without that field.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import os
import warnings
from typing import Any, Awaitable, Callable, Sequence

from ..images import Media, split_prompt
from .base import Backend, BackendError, BackendRequestError, Generation, Sampling
from .http import base_url, new_session, request_json

Transport = Callable[[str, Any], Awaitable[dict]]
BIAS = 100.0
BIDIR_MODES = ("auto", "request", "server", "off")
NEUTRAL = {"temperature": 1.0, "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0, "repeat_penalty": 1.0,
           "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0, "xtc_probability": 0.0}
DEFAULT_MARKER = "<__media__>"      # the media marker of servers that do not report one in /props
COUNT_CACHE = 64                    # text pieces whose token count is remembered (image calls)


def png_data(media: Media) -> list[str]:
    """The request's images as the base64 PNGs llama-server is sent: resized with PIL (bicubic) to the processor
    size where they are not at it yet, and encoded losslessly."""
    from PIL import Image
    resample = getattr(Image, "Resampling", Image).BICUBIC
    out = []
    for img, (h, w) in zip(media.images, media.sizes):
        if (img.height, img.width) != (h, w):
            img = img.resize((w, h), resample)
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=1)
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


class LlamaCppBackend(Backend):
    """See the module docstring. `url`: the llama-server base URL; `bidirectional`: the folder reads its prompt
    bidirectionally (Instant)."""

    name = "llamacpp"

    def __init__(self, url: str, tokenizer: Any, bidirectional: bool = False, timeout: float = 3600.0,
                 attempts: int = 4, retry_delay: float = 1.0, transport: Transport | None = None):
        self.bidir_setting = os.environ.get("STARTLUX_LLAMACPP_BIDIR") or "auto"
        if bidirectional and self.bidir_setting not in BIDIR_MODES:
            raise ValueError("STARTLUX_LLAMACPP_BIDIR must be auto, request, server or off")
        # `auto` is settled by start() from /props; a backend that was not started works as `request`
        self.bidir_mode = "request" if self.bidir_setting == "auto" else self.bidir_setting
        self.url = base_url(url)
        self.tokenizer = tokenizer
        self.bidirectional = bidirectional
        self.timeout, self.attempts, self.retry_delay = timeout, attempts, retry_delay
        self._transport = transport
        self._session = None
        self.reads_images = False               # set by start(): the server was started with an mmproj
        self.media_marker = DEFAULT_MARKER
        self.whole_image_prompt = False         # set by start(): the server reads a prompt with images whole when non-causal
        self._counts: dict[str, int] = {}

    async def start(self) -> None:
        if self._transport is None and self._session is None:
            self._session = new_session(self.timeout, connections=256)
        props = await self._request("/props", None)
        n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
        self.context_length = int(n_ctx) if n_ctx else None
        self.reads_images = bool((props.get("modalities") or {}).get("vision"))
        patch_0003 = bool(props.get("mtmd_prompt_whole_when_non_causal"))
        if self.bidirectional and self.bidir_setting == "auto":
            self.bidir_mode = "server" if patch_0003 else "request"
            logging.info("llamacpp: STARTLUX_LLAMACPP_BIDIR=auto chose %s (/props %s patch 0003)", self.bidir_mode,
                         "reports" if patch_0003 else "does not report")
        if self.reads_images:
            self.media_marker = props.get("media_marker") or DEFAULT_MARKER
            self.whole_image_prompt = patch_0003
            if self.bidirectional and self.bidir_mode == "server" and not self.whole_image_prompt:
                warnings.warn("this llama-server evaluates the text and the images of a prompt in separate micro-batches, "
                              "which cuts the bidirectional attention of an Instant model (read-outs differ from the "
                              "BF16 model's by about 4 times more than with patch 0003, "
                              "0003-server-mtmd-prompt-whole-on-non-causal-context)", RuntimeWarning, stacklevel=2)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def describe(self) -> dict:
        d = {"name": self.name, "server": self.url, "context_length": self.context_length,
             "images": self.reads_images}
        if self.reads_images and self.bidirectional:
            d["image_prompt"] = "whole (one micro-batch)" if self.whole_image_prompt else "split (stock server)"
        if self.bidirectional:
            d["prompt_attention"] = {"request": "bidirectional (per request)", "server": "bidirectional (server)",
                                     "off": "causal (not the trained reading)"}[self.bidir_mode]
            d["bidir_mode"] = self.bidir_mode + (" (auto)" if self.bidir_setting == "auto" else "")
        return d

    def _ids(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    async def readout(self, prompt: str, letter_ids: Sequence[int], media: Media | None = None) -> list[float]:
        if media is not None:
            return await self._media_readout(prompt, letter_ids, media)
        if self.bidirectional and self.bidir_mode != "off":
            return await self.bidirectional_readout(self._ids(prompt), letter_ids)
        return await self._readout(self._ids(prompt), letter_ids)

    async def _media_readout(self, prompt: str, letter_ids: Sequence[int], media: Media) -> list[float]:
        """A read-out over a prompt with images (module docstring)."""
        if self.bidirectional and self.bidir_mode == "request":
            raise BackendRequestError(
                "images with an Instant model need a non-causal llama-server context, which the per-request "
                "bidirectional mode (patch 0001) does not provide: serve a GGUF with qwen35.attention.causal=false "
                "(or --override-kv qwen35.attention.causal=bool:false) with --parallel 1 --ctx-checkpoints 0 and "
                "-b/-ub of at least the longest prompt, and set STARTLUX_LLAMACPP_BIDIR=server")
        field, pieces = await self._media_prompt(prompt, media)
        return await self._readout(field, letter_ids, None, (pieces, media))

    async def _readout(self, ids: Any, letter_ids: Sequence[int], extra: dict | None = None,
                       check: tuple | None = None) -> list[float]:
        """`ids`: the prompt as token ids, or (images) the prompt object with its media; `check`: (text pieces,
        media) of an image call, whose token count is compared with the server's."""
        k = len(letter_ids)
        body = dict(NEUTRAL, prompt=ids, n_predict=1, n_probs=k, post_sampling_probs=True, top_k=k,
                    logit_bias=[[t, BIAS] for t in letter_ids], cache_prompt=False, seed=0, **(extra or {}))
        out = await self._request("/completion", body)
        if check is not None:
            await self._check_prompt_tokens(out, *check)
        if extra and "prompt_bidirectional" in extra:
            echo = (out.get("generation_settings") or {}).get("prompt_bidirectional")
            if echo != extra["prompt_bidirectional"]:
                raise BackendError("llama-server ignored prompt_bidirectional: it needs the patch "
                                   "0001-server-bidirectional-prompts (or set STARTLUX_LLAMACPP_BIDIR=server for a "
                                   "server whose context is non-causal)")
        rows = out.get("completion_probabilities") or out.get("probs") or []
        if not rows:
            raise BackendError("llama-server returned no token probabilities; it needs n_probs support")
        entry = rows[0]
        got = {}
        if "top_probs" in entry:
            for c in entry["top_probs"]:
                got[c["id"]] = math.log(c["prob"]) if c["prob"] > 0 else -1e9
        else:                                       # an older server ignored post_sampling_probs: raw log-probabilities
            for c in entry.get("top_logprobs", []):
                got[c["id"]] = c["logprob"]
        missing = [t for t in letter_ids if t not in got]
        if missing:
            raise BackendError(f"llama-server did not return {len(missing)} of the {k} option letters; it needs "
                               "post_sampling_probs (a recent llama.cpp build)")
        return [got[t] for t in letter_ids]

    async def bidirectional_readout(self, ids: list[int], letter_ids: Sequence[int]) -> list[float]:
        """Letter log-probabilities after reading the prompt `ids` with bidirectional attention in the full-attention
        layers (and causal linear-attention layers); see STARTLUX_LLAMACPP_BIDIR in the module docstring."""
        if self.bidir_mode == "request":
            return await self._readout(ids, letter_ids, {"prompt_bidirectional": -1})
        return await self._readout(ids, letter_ids)

    async def generate(self, prompt: str, max_tokens: int, sampling: Sampling, stop: Sequence[str],
                       seed: int, media: Media | None = None) -> Generation:
        check = None
        if media is None:
            field = self._ids(prompt)
        else:
            field, pieces = await self._media_prompt(prompt, media)
            check = (pieces, media)
        body = dict(NEUTRAL, prompt=field, n_predict=max_tokens, temperature=sampling.temperature,
                    top_p=sampling.top_p, top_k=sampling.top_k, stop=list(stop), seed=seed, cache_prompt=False,
                    samplers=["top_k", "temperature", "top_p"])     # vLLM order: top-p on the tempered distribution
        if sampling.temperature <= 0:
            body.update(samplers=["top_k"], top_k=1)
        out = await self._request("/completion", body)
        if check is not None:
            await self._check_prompt_tokens(out, *check)
        kind = out.get("stop_type")
        if kind is None:                            # older servers: one flag per kind
            kind = "eos" if out.get("stopped_eos") else "word" if out.get("stopped_word") else "limit"
        return Generation(out.get("content", ""), int(out.get("tokens_predicted", 0)),
                          "stop" if kind in ("eos", "word") else "length")

    # ---------------------------------------------------------------- images
    async def _media_prompt(self, prompt: str, media: Media) -> tuple[dict, list[str]]:
        """The prompt object of an image call and the text pieces between the images' places."""
        if not self.reads_images:
            raise BackendRequestError("this llama-server has no multimodal projector: start it with --mmproj FILE "
                                      "(the mmproj-*-F16.gguf of the model)")
        pieces = split_prompt(prompt)
        if len(pieces) != len(media) + 1:
            raise BackendRequestError(f"the prompt has {len(pieces) - 1} image places for {len(media)} images")
        if any(self.media_marker in p for p in pieces):
            raise BackendRequestError("the text contains the server's media marker")
        data = await self._encoded(media)
        return {"prompt_string": self.media_marker.join(pieces), "multimodal_data": data}, pieces

    @staticmethod
    async def _encoded(media: Media) -> list[str]:
        """The request's images as base64 PNGs (png_data), made once in a thread for all calls of the request."""
        done = getattr(media, "_llamacpp_png", None)
        if isinstance(done, list):
            return done
        if done is None:
            done = asyncio.ensure_future(asyncio.to_thread(png_data, media))
            media._llamacpp_png = done
        data = await asyncio.shield(done)
        media._llamacpp_png = data
        return data

    def _count(self, piece: str) -> int:
        n = self._counts.get(piece)
        if n is None:
            if len(self._counts) >= COUNT_CACHE:
                self._counts.clear()
            n = self._counts[piece] = len(self._ids(piece))
        return n

    async def _check_prompt_tokens(self, out: dict, pieces: list[str], media: Media) -> None:
        """The server evaluated the prompt we meant: text tokens + (vision start and end + the image's tokens) per
        image. A server that resized an image (or reads it with another projector) gives another count."""
        got = out.get("tokens_evaluated")
        if not isinstance(got, int):
            return                                      # an older server does not say
        fixed = sum(media.tokens) + 2 * len(media)
        want = sum(self._count(p) for p in pieces) + fixed
        if got == want:
            return
        theirs = fixed
        for p in pieces:                                # llama.cpp's tokenizer may split a rare word differently
            theirs += len((await self._request("/tokenize", {"content": p, "add_special": False,
                                                              "parse_special": True})).get("tokens") or [])
        if got == theirs:
            return
        lo, hi = max(1, media.min_pixels // 1024), max(1, media.max_pixels // 1024)
        raise BackendError(f"llama-server evaluated {got} prompt tokens, the engine expected {want} ({sum(media.tokens)} "
                           f"image tokens in {len(media)} images): the server resized an image or read it with "
                           f"another projector. Start it with the mmproj of this model, --image-min-tokens {lo} "
                           f"--image-max-tokens {hi} (the engine's pixel bounds) and -ub large enough for the prompt")

    async def _request(self, path: str, body: dict | None) -> dict:
        if self._transport is not None:
            return await self._transport(self.url + path, body)
        return await request_json(self._session, self.url + path, body, self.attempts, self.retry_delay)
