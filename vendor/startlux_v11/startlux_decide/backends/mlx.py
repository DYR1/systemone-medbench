"""In-process engine on Apple Silicon with mlx-vlm: text and images, Instant and AutoThink folders.

The forward pass runs in mlx-vlm's implementation of the model (the folder's language model and vision tower); the
package builds the input around it and runs the layers itself, so that the full-attention layers can get the mask a
folder was trained with. Prompts, the letter read-out, temperatures, the gate and the thought loop are the package's own
code, as with every engine.

Folders: an mlx-vlm folder (``mlx_vlm.convert``: the text model quantized, the vision tower and everything else in
BF16), the BF16 release folder itself, or a text-only MLX folder without a vision tower (it reads text only: a request
with images gets a 422).

* Read-out: the hidden state at the last prompt position times the output-head rows of the option letters (float32).
* Instant folders read their prompt bidirectionally in the full-attention layers (every token of the one-pass input
  sees every other token, up to and including the read-out position); the linear-attention layers (Gated DeltaNet and
  its short convolution) stay causal. This is the mask of backends/bidirectional.py, there for transformers. AutoThink
  folders are causal throughout, thoughts included.
* Thoughts (AutoThink): the prompt is read into a cache in chunks, then tokens are sampled one by one with the sampling
  settings of the request (temperature, then top-k, then top-p, by inverse CDF with draws from the request's seed), until
  a stop string, an end-of-sequence token or the token budget.
* Images: the folder's image processor gives the pixel values and the grid of every image (resized within the bounds
  of the request's Media), the vision tower writes its features at the image tokens of the prompt, and the decoder gets
  the three-axis (M-RoPE) positions of the processor: an image's tokens sit at (start, start + row, start + column) of
  its merged grid and the text after it continues at start + max(rows, columns). The features of the last request are
  kept, so the second read-out of a thought (same images) does not run the vision tower again.

Everything runs on one thread (MLX streams belong to the thread that made them); calls queue behind each other.
"""
from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from ..images import IMAGE_PAD
from .base import Backend, BackendRequestError, Generation, Sampling

CACHE_LIMIT_BYTES = 1 << 30           # MLX's buffer cache (freed GPU buffers kept for reuse)
PREFILL_CHUNK = 8192                  # tokens per step when a causal prompt is read into a cache


# ----------------------------------------------------------------------------------------------- pure helpers
def rope_index(ids: Sequence[int], image_token_id: int, grids: Sequence[Sequence[int]], merge: int = 2):
    """(positions [3, L], delta): the three position axes (time, height, width) of every token of a prompt and
    ``max position + 1 - L``, the shift between a token's index and its position after the last image.

    Text tokens (the image markers around an image included) take one position each, the same on all three axes. The
    tokens of an image (one run of `image_token_id` per image, in the order of `grids`, the (t, h, w) of the processor
    in patches) take (start, start + row, start + column) over the merged grid (h // merge by w // merge), and the text
    after the image continues at start + max(rows, columns). Rule of the models' processor (get_rope_index)."""
    import numpy as np

    ids = np.asarray(ids)
    length = len(ids)
    is_image = ids == image_token_id
    pos = np.zeros((3, length), dtype=np.int64)
    cur, i, k = 0, 0, 0
    while i < length:
        j = i
        if not is_image[i]:
            while j < length and not is_image[j]:
                j += 1
            pos[:, i:j] = np.arange(cur, cur + j - i)
            cur += j - i
        else:
            while j < length and is_image[j]:
                j += 1
            if k >= len(grids):
                raise ValueError("the prompt has more images than grids")
            t, h, w = (int(x) for x in grids[k])
            k += 1
            rows, cols = h // merge, w // merge
            if t != 1 or j - i != rows * cols:
                raise ValueError(f"an image of grid {(t, h, w)} takes {rows * cols} tokens, the prompt has {j - i}")
            pos[0, i:j] = cur
            pos[1, i:j] = cur + np.repeat(np.arange(rows), cols)
            pos[2, i:j] = cur + np.tile(np.arange(cols), rows)
            cur += max(rows, cols)
        i = j
    if k != len(grids):
        raise ValueError(f"the prompt has {k} images for {len(grids)} grids")
    return pos, int(pos.max()) + 1 - length if length else 0


def first_stop(text: str, stops: Sequence[str], new_chars: int) -> int | None:
    """Index of the earliest stop string that the last `new_chars` characters of `text` completed or started, else None."""
    best = None
    for s in stops:
        i = text.find(s, max(0, len(text) - new_chars - len(s) + 1))
        if i >= 0 and (best is None or i < best):
            best = i
    return best


class _Detokenizer:
    """Text of generated tokens one at a time: the pending tokens are decoded together and emitted once they no longer
    end in an incomplete UTF-8 sequence."""

    def __init__(self, tokenizer: Any):
        self.tokenizer, self.pending = tokenizer, []

    def add(self, token: int) -> str:
        self.pending.append(token)
        s = self.tokenizer.decode(self.pending, skip_special_tokens=True)
        if s.endswith("�") and len(self.pending) < 8:
            return ""
        self.pending = []
        return s

    def flush(self) -> str:
        s = self.tokenizer.decode(self.pending, skip_special_tokens=True) if self.pending else ""
        self.pending = []
        return s


def sample_tokens(logits: Any, u: Any, temperature: float, top_k: int, top_p: float) -> Any:
    """One token id per row of `logits` [rows, vocab] by inverse CDF with the row's uniform u[row]: temperature first,
    then top-k, then top-p (the smallest set of top tokens whose mass reaches top_p; the top token is always kept), the
    order vLLM applies them. temperature <= 0 is greedy; top_k <= 0 and top_p >= 1 switch those filters off."""
    import mlx.core as mx

    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    x = logits.astype(mx.float32) / temperature
    vocab = x.shape[-1]
    if 0 < top_k < vocab:
        idx = mx.argpartition(-x, kth=top_k - 1, axis=-1)[..., :top_k]
        vals = mx.take_along_axis(x, idx, axis=-1)
    else:
        idx, vals = None, x
    order = mx.argsort(-vals, axis=-1)
    vals = mx.take_along_axis(vals, order, axis=-1)
    idx = order if idx is None else mx.take_along_axis(idx, order, axis=-1)
    probs = mx.softmax(vals, axis=-1)
    if top_p < 1.0:
        above = mx.cumsum(probs, axis=-1) - probs          # mass of the tokens ranked above each token
        probs = mx.where(above < top_p, probs, 0.0)
    cdf = mx.cumsum(probs, axis=-1)
    target = u.astype(mx.float32)[:, None] * cdf[..., -1:]
    j = mx.minimum(mx.sum(cdf < target, axis=-1), probs.shape[-1] - 1)
    return mx.take_along_axis(idx, j[:, None], axis=-1)[:, 0]


# ----------------------------------------------------------------------------------------------- the engine
class MLXBackend(Backend):
    """`path`: an MLX model folder (see the module docstring) with decision_config.json next to it; `bidirectional`: an
    Instant folder (it answers in one pass and does not generate)."""

    name = "mlx"

    def __init__(self, path: str, tokenizer: Any, bidirectional: bool, prefill_chunk: int = PREFILL_CHUNK):
        self.path, self.tokenizer, self.bidirectional = path, tokenizer, bidirectional
        self.prefill_chunk = max(int(prefill_chunk), 1)
        self.config: dict = {}
        self.model = self.lm = self.body = self.vision = self.processor = None
        self.layers: list = []
        self.check: dict | None = None
        self.image_token_id = 248056
        self.merge = 2
        self.eos: set[int] = set()
        self._letters: dict = {}
        self._features: tuple | None = None            # (media, grids, features) of the last request with images
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="startlux-mlx")

    async def start(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._executor, self._load)

    def _load(self) -> None:
        import mlx.core as mx

        try:
            from mlx_vlm.utils import load_model, load_processor
        except ImportError as e:
            raise RuntimeError("the mlx engine needs mlx-vlm (Apple Silicon): pip install mlx-vlm") from e
        folder = Path(self.path)
        self.config = json.loads((folder / "config.json").read_text())
        text = self.config.get("text_config", self.config)
        mx.set_cache_limit(CACHE_LIMIT_BYTES)
        self.model = load_model(folder)
        self.lm = self.model.language_model
        self.body = self.lm.model
        self.layers = list(self.body.layers)
        self.vision = getattr(self.model, "vision_tower", None)
        self.image_token_id = int(self.config.get("image_token_id", 248056))
        if self.vision is not None:
            self.merge = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
            try:
                self.processor = load_processor(folder, add_detokenizer=False)
                self.reads_images = True
            except (OSError, ValueError, ImportError) as e:
                logging.warning("mlx: %s has a vision tower but no usable image processor (%s): text only", folder, e)
                self.processor = None
        self.eos = _eos_ids(self.config, text, folder, self.tokenizer)
        self.check = self._self_check()

    async def close(self) -> None:
        self._executor.shutdown(wait=True)

    def describe(self) -> dict:
        import importlib.metadata as md
        import mlx.core as mx

        def version(name: str) -> str | None:
            try:
                return md.version(name)
            except md.PackageNotFoundError:
                return None

        return {"name": self.name, "prompt_attention": "bidirectional" if self.bidirectional else "causal",
                "images": self.reads_images, "quantization": self.config.get("quantization"),
                "mlx": mx.__version__, "mlx_vlm": version("mlx-vlm"), "prompt_attention_check": self.check}

    # ------------------------------------------------------------------------------------------- forward pass
    def _head(self, h: Any) -> Any:
        """Logits [rows, vocab] from final hidden states [rows, hidden]."""
        if hasattr(self.lm, "lm_head"):
            return self.lm.lm_head(h)
        return self.body.embed_tokens.as_linear(h)

    def _forward(self, embeds: Any, pos: Any, mask: Any, cache: list | None = None) -> Any:
        """Final-norm hidden states [1, L, hidden] of input embeddings [1, L, hidden] with positions pos [3, L].

        `mask` goes to the full-attention layers: None = every token sees every token of the input (bidirectional, or
        a single new token), "causal" = each token sees itself and earlier ones (with a cache: after the cached ones).
        The linear-attention layers are causal by construction and take no mask (one row, no padding)."""
        import mlx.core as mx
        import numpy as np

        positions = mx.array(np.asarray(pos, dtype=np.int32).reshape(3, 1, -1))
        h, shared = embeds, None
        first = next((layer for layer in self.layers if not layer.is_linear), None)
        if first is not None and not first.self_attn.rotary_emb.fused_apply:
            shared = first.self_attn.rotary_emb(h, positions)          # cos and sin, once for all layers
        for layer, c in zip(self.layers, cache if cache is not None else [None] * len(self.layers)):
            if layer.is_linear:
                h = layer(h, mask=None, cache=c)
            else:
                h = layer(h, mask=mask, cache=c, position_ids=positions, position_embeddings=shared)
        return self.body.norm(h)

    def _self_check(self, tokens: int = 24) -> dict:
        """Check on the loaded model that the folder's prompt attention is in effect: a change of the last token of a
        row must reach the first position of a bidirectional model and must not reach it in a causal one."""
        import mlx.core as mx
        import numpy as np

        vocab = int(self.body.embed_tokens.weight.shape[0])
        base = np.random.default_rng(1234).integers(min(1000, vocab // 4), min(50000, vocab - 1), size=tokens).astype(np.int32)
        late = base.copy()
        late[-1] += 1
        pos = np.broadcast_to(np.arange(tokens), (3, tokens))
        mask = None if self.bidirectional else "causal"
        h = [self._forward(self.body.embed_tokens(mx.array(x)[None]), pos, mask)[0, 0].astype(mx.float32)
             for x in (base, late)]
        reach = float(mx.abs(h[0] - h[1]).max())
        if self.bidirectional != (reach > 0):
            raise RuntimeError(f"prompt attention {'bidirectional' if self.bidirectional else 'causal'} is not in "
                               f"effect on the loaded model (first-position change {reach})")
        return {"first_position_change": reach, "full_attention_layers": sum(not l.is_linear for l in self.layers),
                "layers": len(self.layers)}

    # ------------------------------------------------------------------------------------------- inputs
    def _image_features(self, media: Any) -> tuple:
        """(grids [n, 3], features [tokens, hidden]) of the images of a request; the last request's are kept."""
        import mlx.core as mx
        import numpy as np

        if self._features is not None and self._features[0] is media:
            return self._features[1], self._features[2]
        out = self.processor.image_processor(images=list(media.images), min_pixels=media.min_pixels,
                                             max_pixels=media.max_pixels)
        grids = np.asarray(out["image_grid_thw"], dtype=np.int64).reshape(-1, 3)
        tokens = [int(g.prod()) // self.merge ** 2 for g in grids]
        if tokens != list(media.tokens):
            raise RuntimeError(f"the processor gave {tokens} image tokens, the package counted {list(media.tokens)}")
        dtype = self.vision.patch_embed.proj.weight.dtype
        feats = self.vision(mx.array(out["pixel_values"]).astype(dtype), mx.array(grids.astype(np.int32)))[0]
        mx.eval(feats)
        self._features = (media, grids, feats)
        return grids, feats

    def _inputs(self, prompt: str, media: Any) -> tuple:
        """(token ids [L], embeddings [1, L, hidden], positions [3, L], delta) of a prompt, its image placeholders
        replaced by the vision features."""
        import mlx.core as mx
        import numpy as np

        if media is None:
            ids = np.asarray(self.tokenizer.encode(prompt, add_special_tokens=False), dtype=np.int32)
            return (ids, self.body.embed_tokens(mx.array(ids)[None]),
                    np.broadcast_to(np.arange(len(ids)), (3, len(ids))), 0)
        if not self.reads_images:
            raise BackendRequestError("this model folder has no vision tower or image processor: it reads text only")
        grids, feats = self._image_features(media)
        parts = prompt.split(IMAGE_PAD)
        if len(parts) - 1 != len(grids):
            raise RuntimeError(f"the prompt has {len(parts) - 1} image placeholders for {len(grids)} images")
        text = parts[0]
        for g, rest in zip(grids, parts[1:]):
            text += IMAGE_PAD * (int(g.prod()) // self.merge ** 2) + rest
        ids = np.asarray(self.tokenizer.encode(text, add_special_tokens=False), dtype=np.int32)
        pos, delta = rope_index(ids, self.image_token_id, grids, self.merge)
        slots = ids == self.image_token_id
        embeds = self.body.embed_tokens(mx.array(ids)[None])
        rows = mx.array(np.where(slots, np.cumsum(slots) - 1, 0).astype(np.int32))
        embeds = mx.where(mx.array(slots)[None, :, None], feats[rows][None].astype(embeds.dtype), embeds)
        return ids, embeds, pos, delta

    def _letter_rows(self, letter_ids: Sequence[int]) -> Any:
        """The output-head rows of the letters, float32 (a quantized head is dequantized for these rows only)."""
        import mlx.core as mx

        key = tuple(int(t) for t in letter_ids)
        w = self._letters.get(key)
        if w is None:
            head = self.lm.lm_head if hasattr(self.lm, "lm_head") else self.body.embed_tokens
            idx = mx.array(key)
            w = head.weight[idx]
            if hasattr(head, "scales"):
                biases = head.biases[idx] if getattr(head, "biases", None) is not None else None
                w = mx.dequantize(w, head.scales[idx], biases, head.group_size, head.bits,
                                  mode=getattr(head, "mode", "affine"))
            w = w.astype(mx.float32)
            mx.eval(w)
            self._letters[key] = w
        return w

    # ------------------------------------------------------------------------------------------- read-out
    async def readout(self, prompt: str, letter_ids: Sequence[int], media: Any = None) -> list[float]:
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._readout, prompt,
                                                                list(letter_ids), media)

    def _readout(self, prompt: str, letter_ids: list[int], media: Any = None) -> list[float]:
        import mlx.core as mx

        ids, embeds, pos, _ = self._inputs(prompt, media)
        if self.bidirectional or len(ids) <= self.prefill_chunk:
            h = self._forward(embeds, pos, None if self.bidirectional else "causal")[:, -1]
        else:                                                        # a long causal prompt, a chunk at a time
            h = self._prefill(embeds, pos, self.lm.make_cache())
        z = h.astype(mx.float32) @ self._letter_rows(letter_ids).T
        return z[0].tolist()

    def _prefill(self, embeds: Any, pos: Any, cache: list) -> Any:
        """Read a causal prompt into `cache`, chunk by chunk; returns the final-norm hidden state of the last token."""
        import mlx.core as mx

        n = embeds.shape[1]
        for a in range(0, n, self.prefill_chunk):
            b = min(a + self.prefill_chunk, n)
            h = self._forward(embeds[:, a:b], pos[:, a:b], "causal" if b - a > 1 else None, cache)
            mx.eval(h, [c.state for c in cache])
        return h[:, -1]

    # ------------------------------------------------------------------------------------------- thoughts
    async def generate(self, prompt: str, max_tokens: int, sampling: Sampling, stop: Sequence[str],
                       seed: int, media: Any = None) -> Generation:
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._generate, prompt, max_tokens, sampling, list(stop), seed, media)

    def _generate(self, prompt: str, max_tokens: int, sampling: Sampling, stop: list[str], seed: int,
                  media: Any = None) -> Generation:
        import mlx.core as mx
        import numpy as np

        if self.bidirectional:
            raise NotImplementedError("Instant (prompt-bidirectional) folders answer in one pass and do not generate")
        steps = int(max_tokens)
        if steps <= 0:
            return Generation("", 0, "length")
        ids, embeds, pos, delta = self._inputs(prompt, media)
        n = len(ids)
        stop = [s for s in stop if s]
        single = {}                                   # stop strings that are exactly one token
        for s in stop:
            t = self.tokenizer.encode(s, add_special_tokens=False)
            if len(t) == 1:
                single[t[0]] = s
        multi = [s for s in stop if s not in single.values()]
        draws = mx.random.uniform(shape=(steps,), key=mx.random.key(int(seed) % (2 ** 63)))

        def pick(h: Any, k: int) -> Any:
            return sample_tokens(self._head(h), draws[k:k + 1], sampling.temperature, sampling.top_k, sampling.top_p)

        cache = self.lm.make_cache()
        y = pick(self._prefill(embeds, pos, cache), 0)
        mx.async_eval(y)
        text, tokens, finish = "", 0, "length"
        detok = _Detokenizer(self.tokenizer)
        for k in range(steps):
            nxt = None
            if k + 1 < steps:                          # the next step runs on the device while this one is handled
                p = np.full((3, 1), n + k + delta)
                h = self._forward(self.body.embed_tokens(y.reshape(1, 1)), p, None, cache)[:, -1]
                nxt = pick(h, k + 1)
                mx.async_eval(nxt)
            tok = int(y.item())
            tokens += 1
            if tok in single or tok in self.eos:
                finish = "stop"
                break
            piece = detok.add(tok)
            text += piece
            if multi and piece:
                hit = first_stop(text, multi, len(piece))
                if hit is not None:
                    text, finish = text[:hit], "stop"
                    break
            if nxt is None:
                break
            y = nxt
        if finish == "length":
            text += detok.flush()
        return Generation(text, tokens, finish)


def _eos_ids(config: dict, text_config: dict, folder: Path, tokenizer: Any) -> set[int]:
    """The tokens that end a continuation: the end-of-sequence ids of config.json and generation_config.json and the
    tokenizer's own (the end of an assistant turn)."""
    found: list = [config.get("eos_token_id"), text_config.get("eos_token_id"), getattr(tokenizer, "eos_token_id", None)]
    gen = folder / "generation_config.json"
    if gen.is_file():
        found.append(json.loads(gen.read_text()).get("eos_token_id"))
    out: set[int] = set()
    for v in found:
        for t in (v if isinstance(v, (list, tuple)) else [v]):
            if t is not None:
                out.add(int(t))
    return out
