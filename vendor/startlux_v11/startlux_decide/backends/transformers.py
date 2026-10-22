"""In-process reference engine on PyTorch and transformers (one prompt per forward pass, calls run one at a time).

Slow next to vLLM and meant for checking: it reads Instant folders with the bidirectional prompt attention they were
trained with (bidirectional.py) and runs AutoThink folders causally, thoughts included (transformers' sampler, so a
thought is not token-identical to vLLM's for the same seed). On CUDA, install flash-linear-attention and
causal-conv1d: without them transformers runs the linear-attention layers on a plain PyTorch path that is more than
ten times slower.

Images: when the folder has a vision tower, a prompt with image placeholders goes through the folder's processor
(the images resized within the request's bounds, each placeholder expanded to the image's tokens), the vision tower
writes its features at those tokens, and the decoder reads them with the multimodal positions (get_rope_index), as
the model's own forward does; Instant folders read that input with bidirectional prompt attention.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from .base import Backend, BackendRequestError, Generation, Sampling


class TransformersBackend(Backend):
    """`path`: the model folder; `bidirectional`: an Instant folder; `device`: "cuda", "cpu", ... (default: cuda when
    available); `dtype`: "bfloat16" (the released weights) or "float32"."""

    name = "transformers"

    def __init__(self, path: str, tokenizer: Any, bidirectional: bool, device: str | None = None,
                 dtype: str = "bfloat16"):
        self.path, self.tokenizer, self.bidirectional = path, tokenizer, bidirectional
        self.device_name, self.dtype_name = device, dtype
        self.model = self.body = self.head = self.device = self.processor = None
        self.check: dict | None = None
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="startlux-transformers")

    async def start(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._executor, self._load)

    def _load(self) -> None:
        import torch
        import transformers

        config = transformers.AutoConfig.from_pretrained(self.path)
        cls = getattr(transformers, config.architectures[0])
        self.device = torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = cls.from_pretrained(self.path, dtype=getattr(torch, self.dtype_name),
                                         device_map={"": self.device}).eval()
        inner = self.model.model
        self.body = getattr(inner, "language_model", inner)
        self.head = self.model.get_output_embeddings().weight
        if getattr(inner, "visual", None) is not None:
            try:
                self.processor = transformers.AutoProcessor.from_pretrained(self.path)
                self.reads_images = True
            except (OSError, ValueError, ImportError):
                self.processor = None
        if self.bidirectional:
            from . import bidirectional
            self.check = bidirectional.enable(self.body)

    async def close(self) -> None:
        self._executor.shutdown(wait=True)

    def describe(self) -> dict:
        return {"name": self.name, "device": str(self.device), "dtype": self.dtype_name,
                "prompt_attention": "bidirectional" if self.bidirectional else "causal",
                "images": self.reads_images, "bidirectional_check": self.check}

    def _inputs(self, prompt: str, media: Any) -> dict:
        """The model inputs of a prompt: its token ids, and with images the processor's pixel values, image grids
        and token types (each image placeholder expanded to the image's tokens)."""
        import torch

        if media is None:
            return {"input_ids": torch.tensor([self.tokenizer.encode(prompt, add_special_tokens=False)],
                                              device=self.device)}
        if self.processor is None:
            raise BackendRequestError("this model folder has no image processor: it reads text only")
        enc = self.processor(text=[prompt], images=list(media.images), size=media.processor_size, return_tensors="pt")
        ids = enc["input_ids"]
        image_token = self.model.config.image_token_id
        if int((ids == image_token).sum()) != sum(media.tokens):
            raise RuntimeError(f"the processor gave {int((ids == image_token).sum())} image tokens, the package "
                               f"counted {sum(media.tokens)}")
        out = {k: enc[k].to(self.device) for k in ("input_ids", "pixel_values", "image_grid_thw") if k in enc}
        types = enc.get("mm_token_type_ids")
        out["mm_token_type_ids"] = (types if types is not None else (ids == image_token).int()).to(self.device)
        return out

    async def readout(self, prompt: str, letter_ids: Sequence[int], media: Any = None) -> list[float]:
        return await asyncio.get_running_loop().run_in_executor(self._executor, self._readout, prompt,
                                                                list(letter_ids), media)

    def _readout(self, prompt: str, letter_ids: list[int], media: Any = None) -> list[float]:
        import torch

        enc = self._inputs(prompt, media)
        ids = enc["input_ids"]
        with torch.inference_mode():
            if media is not None:
                hidden = self._image_hidden(enc)
            elif self.bidirectional:
                from . import bidirectional
                hidden = bidirectional.text_forward(self.body, ids)
            else:
                hidden = self.body(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
            rows = self.head.index_select(0, torch.tensor(letter_ids, device=self.head.device))
            return (hidden[0, -1].float() @ rows.float().T).tolist()

    def _image_hidden(self, enc: dict) -> Any:
        """Last hidden states of a prompt with images: the model's own forward (causal folders), or its embedding,
        image features and multimodal positions read with bidirectional prompt attention (Instant folders)."""
        import torch

        inner = self.model.model
        if not self.bidirectional:
            return inner(**enc, use_cache=False, return_dict=True).last_hidden_state
        from . import bidirectional
        ids = enc["input_ids"]
        embeds = inner.get_input_embeddings()(ids)
        feats = inner.get_image_features(enc["pixel_values"], enc["image_grid_thw"], return_dict=True).pooler_output
        feats = torch.cat(list(feats), 0).to(embeds.device, embeds.dtype)
        mask, _ = inner.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=feats)
        embeds = embeds.masked_scatter(mask, feats)
        positions, _ = inner.get_rope_index(ids, image_grid_thw=enc["image_grid_thw"], attention_mask=None,
                                            mm_token_type_ids=enc["mm_token_type_ids"])
        return bidirectional.text_forward(self.body, inputs_embeds=embeds, position_ids=positions)

    async def generate(self, prompt: str, max_tokens: int, sampling: Sampling, stop: Sequence[str],
                       seed: int, media: Any = None) -> Generation:
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._generate, prompt, max_tokens, sampling, list(stop), seed, media)

    def _generate(self, prompt: str, max_tokens: int, sampling: Sampling, stop: list[str], seed: int,
                  media: Any = None) -> Generation:
        import torch

        if self.bidirectional:
            raise NotImplementedError("Instant (prompt-bidirectional) folders answer in one pass and do not generate")
        enc = self._inputs(prompt, media)
        x = enc.pop("input_ids")
        ids = x[0].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos] if eos is not None else [])
        torch.manual_seed(seed)
        with torch.inference_mode():
            out = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_tokens,
                                      do_sample=True, temperature=sampling.temperature, top_p=sampling.top_p,
                                      top_k=sampling.top_k, pad_token_id=self.tokenizer.pad_token_id,
                                      stopping_criteria=_stop_on_text(self.tokenizer, len(ids), stop), **enc)
        new = out[0, len(ids):].tolist()
        tokens, finish = len(new), "length"
        if new and new[-1] in eos:
            new, finish = new[:-1], "stop"
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        for s in stop:
            if s in text:
                text, finish = text[:text.index(s)], "stop"
        return Generation(text, tokens, finish)


def _stop_on_text(tokenizer: Any, start: int, stops: list[str]) -> Any:
    """Stopping criteria that end generation once the decoded tail of the new tokens contains a stop string (works
    with any tokenizer that decodes; the stop strings here are a few tokens long)."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnText(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            new = input_ids[0, max(start, input_ids.shape[1] - 32):].tolist()
            tail = tokenizer.decode(new, skip_special_tokens=True)
            return torch.full((input_ids.shape[0],), any(s in tail for s in stops), dtype=torch.bool,
                              device=input_ids.device)

    return StoppingCriteriaList([StopOnText()] if stops else [])
