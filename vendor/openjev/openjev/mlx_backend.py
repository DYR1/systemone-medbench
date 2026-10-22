"""Structured reads on Apple silicon: DiffusionGemma in-process through MLX.

Selected with OPENJEV_BACKEND=mlx. The read (prefill, one decoder pass, no
self-conditioning, temperature-1 log-softmax per slot) matches what vLLM's
read-only step reports. mlx, mlx_vlm and PIL are imported lazily so the vLLM
path never needs them.

Images go through the processor's chat template and prefill as pixel values;
the decoder pass is the same either way.

Text generation (a thought before a read, and /v1/chat/completions) runs
mlx_vlm's own stream_diffusion_generate rather than a denoise loop of our own:
canvas sizing, self-conditioning and the confidence-threshold unmasking
schedule are the model's published generation policy, and reimplementing them
would only be a slower way to disagree with the checkpoint. It also detokenizes,
which is why dropping the thought-channel markers is asked of it rather than
done to the text it hands back.
"""
import asyncio
import base64
import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from .engine import TOPK, Engine, SchemaError, slot_distribution

# Re-reads and samples repeat a prompt exactly, so its prefill is cached, evicted by a
# token budget (not entry count) so a few long prompts can't pin memory.
PROMPT_CACHE_TOKENS = 16384


class ImagePrompt:
    """An image read, as everything the runtime thread needs to build it: the
    system and state text, and each image's data URL. The expansion into image
    tokens is the processor's, so it cannot be done here, off that thread."""

    __slots__ = ("sys_text", "state_text", "images", "key")

    def __init__(self, sys_text, state_text, images):
        self.sys_text = sys_text
        self.state_text = state_text
        self.images = images
        # the digests keep a cached prefill off a different image, and off the same text with none
        digests = tuple(hashlib.sha256(u.encode()).digest() for u in images)
        self.key = (sys_text, state_text, digests)

    def pil(self):
        from io import BytesIO

        from PIL import Image

        return [Image.open(BytesIO(base64.b64decode(u.partition(",")[2]))).convert("RGB")
                for u in self.images]


class MlxRuntime:
    """The model and the one thread that touches it. MLX work is kept off the
    event loop and on a single thread, from loading on."""

    def __init__(self, model_path):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openjev-mlx")
        self.prefills = OrderedDict()
        self.pool.submit(self._load, model_path).result()

    def _load(self, model_path):
        import mlx.core as mx
        from mlx_vlm import load

        self.mx = mx
        self.model, self.processor = load(model_path, trust_remote_code=False)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def _inputs(self, prompt):
        """(cache key, prefill kwargs, prompt length in tokens) for a read."""
        if not isinstance(prompt, ImagePrompt):
            return tuple(prompt), {"input_ids": self.mx.array([prompt])}, len(prompt)
        from mlx_vlm.utils import prepare_inputs

        images = prompt.pil()
        # this model's message format is LIST_WITH_IMAGE_TYPE_TEXT, so the placeholders
        # the processor later expands must come from structured content, images first
        text = self.processor.apply_chat_template(
            [{"role": "system", "content": prompt.sys_text},
             {"role": "user", "content": [{"type": "image"}] * len(images)
                                         + [{"type": "text", "text": prompt.state_text}]}],
            add_generation_prompt=True, tokenize=False)
        inputs = prepare_inputs(self.processor, images=images, prompts=text)
        ids = inputs["input_ids"]
        kwargs = {"input_ids": ids, "pixel_values": inputs.get("pixel_values"),
                  "mm_token_type_ids": inputs.get("mm_token_type_ids"),
                  "attention_mask": inputs.get("attention_mask")}
        return prompt.key, kwargs, int(ids.shape[-1])

    def _prefill(self, prompt, max_tokens):
        key, kwargs, n = self._inputs(prompt)
        # an image prompt's length is only known here, after the processor expanded it
        if n > max_tokens:
            raise SchemaError(f"the request is {n} tokens; the limit is {max_tokens}")
        hit = self.prefills.get(key)
        if hit is None:
            cache = self.model.diffusion_prefill_cache(**kwargs)
            self.prefills[key] = (cache, n)
            while len(self.prefills) > 1 and sum(t for _, t in self.prefills.values()) > PROMPT_CACHE_TOKENS:
                self.prefills.popitem(last=False)
        else:
            cache = hit[0]
            self.prefills.move_to_end(key)
        return cache, n

    def read(self, prompt, canvas, slots, max_tokens, steps=1):
        """(logprobs at each slot, prompt tokens). The logprobs are {token id:
        logprob} for the top-k tokens and every label. The token count is the
        expanded prompt, image tokens included. Runs on the runtime's thread.

        ``steps`` denoise passes share one prefill and one mask mapping, so more
        steps cost GPU time but not prompt tokens. Between passes only the slot
        positions are written back: vLLM pins the rest of the canvas through
        diffusion_pinned, and here the template simply is never overwritten, so
        it cannot drift. The logprobs returned are the last pass's."""
        mx = self.mx
        cache, n = self._prefill(prompt, max_tokens)
        ids = mx.array([canvas])
        masks = self.model.diffusion_decoder_masks(ids, cache, None)
        pos = mx.array([s["pos"] for s in slots])
        sc, sc_ctx = None, None
        for step in range(steps):
            logits = self.model.diffusion_decoder_logits(ids, cache=cache, self_conditioning=sc,
                                                         decoder_attention_mask=masks)
            if step + 1 == steps:
                break
            # argmax over the slot rows alone; the other ~85% of the canvas is pinned
            ids[0, pos] = mx.argmax(logits[0, pos], axis=-1).astype(ids.dtype)
            if sc_ctx is None:
                sc_ctx = self.model.diffusion_prepare_self_conditioning()
            sc = self.model.diffusion_self_conditioning(logits, sc_ctx)
            mx.eval(ids, sc)
        out = []
        for s in slots:
            row = logits[0, s["pos"]].astype(mx.float32)
            lp = row - mx.logsumexp(row)
            keep = sorted(set(mx.argpartition(-lp, TOPK)[:TOPK].tolist()) | set(s["label_ids"]))
            out.append(dict(zip(keep, lp[mx.array(keep)].tolist())))
        return out, n

    def generate(self, prompt, max_tokens, stop_ids, emit, skip_special=None):
        """Greedy generation from ``prompt`` token ids. ``emit(text, token)`` is
        called per token and returns False to stop early (a disconnected client);
        ``stop_ids`` end the reply in addition to the model's own EOS. Returns
        (generated token ids, prompt tokens, finish reason). Runs on the runtime's
        thread and holds it for the whole reply.

        ``skip_special`` are token ids to leave out of the text. A chat reply passes
        the thought-channel markers, because the model opens a channel of its own
        accord on some replies even though the prompt already seeds an empty one,
        and the caller asked for the reply, not the markers. They cannot be filtered
        here: the detokenizer buffers them and flushes them fused into a later
        token's text, so they never arrive as a result of their own. It drops them
        before they enter that buffer, which is why the skipping is its job.
        ``think`` passes nothing, because a thought is exactly what it wants."""
        from mlx_vlm.generate.diffusion import stream_diffusion_generate

        tok = getattr(self.processor, "tokenizer", self.processor)
        crit = tok.stopping_criteria
        # the criteria object lives on the shared processor, so the extra stops are
        # scoped to this call; only one generation runs on this thread at a time
        saved = list(crit.eos_token_ids)
        crit.eos_token_ids = saved + [i for i in (stop_ids or ()) if i not in saved]
        ids, finish = [], "length"
        stream = stream_diffusion_generate(
            self.model, self.processor, tok, self.mx.array([prompt]), None, None,
            max_tokens=max_tokens, skip_special_token_ids=list(skip_special or ()),
            temperature=0.0)
        try:
            for r in stream:
                if r.is_draft or r.diffusion_block_complete:
                    continue
                if r.finish_reason is not None:
                    # the terminal result carries the detokenizer's last buffered
                    # segment, which no earlier result emitted; its token is the
                    # stop token, so the text is kept and the id is not
                    finish = r.finish_reason
                    if r.text:
                        emit(r.text, None)
                    break
                ids.append(int(r.token))
                if not emit(r.text, int(r.token)):
                    finish = "cancelled"
                    break
        finally:
            stream.close()
            crit.eos_token_ids = saved
        return ids, len(prompt), finish


class MlxEngine(Engine):
    def __init__(self, settings, tokenizer):
        super().__init__(settings, tokenizer)
        self.runtime = MlxRuntime(settings.mlx_model)

    async def close(self):
        await super().close()
        self.runtime.close()

    async def think(self, sys_text, state_text, budget):
        """Engine.think on MLX: same (prefix ids, thought tokens, prompt
        tokens) contract, so read_group and _sequential are unchanged. The
        thought pass reads the whole input, which is what gets billed here;
        the read after it bills the input plus the thought separately."""
        prompt = self.chat_prompt_ids(sys_text, state_text, thinking=True) + self.thought_open
        if len(prompt) > self.s.mlx_max_prompt:
            raise SchemaError(f"the request is {len(prompt)} tokens; the limit is {self.s.mlx_max_prompt}")
        async with self.slots:
            ids, billed, _ = await asyncio.get_running_loop().run_in_executor(
                self.runtime.pool, self.runtime.generate, prompt, budget, self.thought_close,
                lambda text, token: True)
        if self.thought_close[0] in ids:
            ids = ids[: ids.index(self.thought_close[0])]
        return prompt + ids + self.thought_close, len(ids), billed

    async def one_read(self, template, slots, sys_text, content, seed, steps=1, prefix=None):
        if isinstance(content, list):
            # images and think/sequential are mutually exclusive, so prefix is None here
            *parts, state = content
            prompt = ImagePrompt(sys_text, state["text"], [p["image_url"]["url"] for p in parts])
        else:
            prompt = prefix if prefix is not None else self.chat_prompt_ids(sys_text, content)
            if len(prompt) > self.s.mlx_max_prompt:
                raise SchemaError(f"the request is {len(prompt)} tokens; the limit is {self.s.mlx_max_prompt}")
        canvas = self.build_canvas(template, slots, seed)
        async with self.slots:
            tops, billed = await asyncio.get_running_loop().run_in_executor(
                self.runtime.pool, self.runtime.read, prompt, canvas, slots, self.s.mlx_max_prompt, steps)
        return [slot_distribution(top, s["label_ids"]) for top, s in zip(tops, slots)], billed
