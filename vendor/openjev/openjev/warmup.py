"""Warm vLLM before the API opens: python -m openjev.warmup

vLLM compiles DiffusionGemma's sampler step per shape the first time it sees
one (batch of one or many, narrow or full-width tile, one step or several),
and the vision tower on its first image. Each costs seconds, so without this
the first users after a restart would wait for them. The entrypoint runs it
once vLLM is ready; a failure is logged and never stops the server starting.
"""
import asyncio
import base64
import io
import logging
import os
import sys
import time

from .chat import Generator
from .config import Settings
from .engine import Engine

log = logging.getLogger("openjev.warmup")

STATE = "Checkout has been down for every customer since 9:02 and we are losing orders."


def questions(n):
    kinds = [
        {"type": "noul", "instructions": "The customer needs a reply within the hour"},
        {"type": "choice", "criteria": {"outage": None, "billing": None, "feature": None}},
        {"type": "score", "instructions": "How upset is the customer", "criteria": ["calm", "annoyed", "furious"]},
    ]
    return {f"q{i}": kinds[i % 3] for i in range(n)}


def sample_image():
    """A 640x480 JPEG data URL, or None without Pillow."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    img = Image.new("RGB", (640, 480), (200, 170, 120))
    ImageDraw.Draw(img).ellipse((120, 140, 520, 340), fill=(170, 60, 40))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def chat(gen, **body):
    upstream, _ = gen.normalize({"model": "diffusiongemma-26b",
                                 "messages": [{"role": "user", "content": "Name three rivers in Europe."}], **body})
    if upstream.get("stream"):
        async with gen.client.stream("POST", "/v1/chat/completions", json=upstream) as r:
            r.raise_for_status()
            async for _ in r.aiter_lines():
                pass
    else:
        (await gen.client.post("/v1/chat/completions", json=upstream)).raise_for_status()


async def warm(engine, gen):
    image = sample_image()
    img_parts = [{"type": "image_url", "image_url": {"url": image}}] if image else None

    def read(n, seed, **opts):
        return engine.decide(questions(n), STATE, seed, images=opts.pop("images", None), options=opts)

    rounds = [
        ("single reads", lambda: [read(1, 1)]),
        ("", lambda: [read(12, 2)]),
        ("", lambda: [read(2, 3, steps=2)]),
        ("", lambda: [read(12, 4, steps=2)]),
        ("image read", lambda: [read(2, 5, images=img_parts)] if img_parts else []),
        ("", lambda: [read(2, 6, images=img_parts, steps=2)] if img_parts else []),
        ("think", lambda: [read(1, 7, think=32)]),
        ("generation", lambda: [chat(gen, max_tokens=64)]),
        ("", lambda: [chat(gen, max_tokens=64, stream=True, logprobs=True, top_logprobs=5)]),
        ("mixed batch", lambda: [read(1, 11), read(3, 12), read(12, 13, steps=2), read(20, 14),
                                 read(2, 15, samples=4), chat(gen, max_tokens=128),
                                 chat(gen, max_tokens=64, logprobs=True, top_logprobs=5)]
                                + ([read(1, 16, images=img_parts)] if img_parts else [])),
        ("", lambda: [read(1, 21, steps=3), read(1, 22, steps=3), chat(gen, max_tokens=64), chat(gen, max_tokens=64)]),
    ]
    start = time.perf_counter()
    for label, make in rounds:
        t = time.perf_counter()
        results = await asyncio.gather(*make(), return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        for e in errors:
            log.warning("warmup request failed: %s: %s", type(e).__name__, e)
        if label:
            log.info("warmup %s: %.1fs", label, time.perf_counter() - t)
    log.info("warmup done in %.1fs", time.perf_counter() - start)


async def main():
    from transformers import AutoTokenizer

    settings = Settings()
    engine = Engine(settings, AutoTokenizer.from_pretrained(settings.tokenizer))
    gen = Generator(settings)
    try:
        await warm(engine, gen)
    finally:
        await engine.close()
        await gen.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="openjev: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if os.environ.get("OPENJEV_WARMUP", "1") == "0":
        sys.exit(0)
    try:
        asyncio.run(main())
    except Exception as e:  # noqa: BLE001  (never keep the server from starting)
        log.warning("warmup skipped: %s: %s", type(e).__name__, e)
