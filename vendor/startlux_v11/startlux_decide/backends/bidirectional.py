"""Bidirectional prompt attention for the one-pass read-out of Instant models, in transformers.

Instant folders were trained to read their prompt bidirectionally:

* full-attention layers (gated attention, every 4th layer): every token of a row attends to every token of the same
  row, earlier and later, over the whole one-pass input (system turn, user turn, assistant header and the empty think
  block, up to and including the read-out position, the last token);
* linear-attention layers (Gated DeltaNet and its short causal convolution): unchanged, causal, given exactly the mask
  the stock forward gives them.

Rows are right-padded; each row attends over its own tokens only (rows with padding are attended one by one at their
exact length, without a mask; a batch without padding runs as one call with is_causal=False). Padded query positions
output zeros; nothing valid reads them.

``text_forward`` follows the forward of the models' text decoder in transformers 5.8 (embeddings, text and multimodal
position ids, rotary embeddings, decoder layers, final norm). ``enable`` routes the full-attention layers of a model
through ``_attention`` and checks, on the model itself, that (1) the stock causal forward is unchanged and equal to the
layer loop with the causal mask, (2) changing padding tokens changes no token state, (3) a late token reaches the
first position.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F

IMPL = "startlux_prompt_bidirectional"


def check_body(body: Any) -> dict:
    """The text decoder must be the transformers 5.8 hybrid decoder this module reproduces, with SDPA attention."""
    import transformers
    if not transformers.__version__.startswith("5.8."):
        raise RuntimeError(f"bidirectional prompt attention is written against transformers 5.8.x, found "
                           f"{transformers.__version__}")
    cfg = body.config
    if cfg._attn_implementation != "sdpa":
        raise ValueError(f"bidirectional prompt attention needs sdpa attention, not {cfg._attn_implementation!r}")
    for name in ("embed_tokens", "rotary_emb", "layers", "norm", "_update_linear_attn_mask"):
        if not hasattr(body, name):
            raise ValueError(f"the text decoder lacks {name}: not the hybrid decoder of these models")
    types = list(cfg.layer_types[:cfg.num_hidden_layers])
    if "full_attention" not in types or set(types) - {"full_attention", "linear_attention"}:
        raise ValueError(f"unexpected layer types {sorted(set(types))}")
    for i, layer in enumerate(body.layers[:cfg.num_hidden_layers]):
        if layer.layer_type != types[i] or (types[i] == "full_attention") != hasattr(layer, "self_attn"):
            raise ValueError(f"layer {i} does not match its declared type {types[i]}")
    return {"full_attention_layers": [i for i, t in enumerate(types) if t == "full_attention"], "layers": len(types)}


def _attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    """Attention of the full-attention layers of an enabled model. Without row lengths: the stock SDPA attention.
    With row lengths (from text_forward): no causal term; each row attends over its first n tokens."""
    lengths = kwargs.pop("startlux_row_lengths", None)
    if lengths is None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      is_causal=is_causal, **kwargs)
    if attention_mask is not None:
        raise ValueError("bidirectional rows take no attention mask")
    rows, heads, length, _ = query.shape
    if len(lengths) != rows:
        raise ValueError("one length per row required")
    gqa = key.shape[1] != heads
    if all(n == length for n in lengths):
        out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout, is_causal=False, scale=scaling,
                                             enable_gqa=gqa)
        return out.transpose(1, 2).contiguous(), None
    out = query.new_zeros(rows, length, heads, query.shape[-1])         # the [B, L, H, D] layout the caller expects
    for i, n in enumerate(lengths):
        o = F.scaled_dot_product_attention(query[i:i + 1, :, :n], key[i:i + 1, :, :n], value[i:i + 1, :, :n],
                                           dropout_p=dropout, is_causal=False, scale=scaling, enable_gqa=gqa)
        out[i, :n] = o[0].transpose(0, 1)
    return out, None


def _kernel_guard(device: torch.device):
    """On CUDA only the fused kernels (flash, memory-efficient): the math kernel would build H x L x L scores."""
    if device.type != "cuda":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])


def _full_layers(body: Any) -> list:
    cfg = body.config
    return [layer for i, layer in enumerate(body.layers[:cfg.num_hidden_layers])
            if cfg.layer_types[i] == "full_attention"]


def text_forward(body: Any, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None,
                 valid: torch.Tensor | None = None, causal: bool = False, inputs_embeds: torch.Tensor | None = None,
                 position_ids: torch.Tensor | None = None) -> torch.Tensor:
    """last_hidden_state [B, L, H] of the text decoder.

    attention_mask: what a caller passes the stock forward (None, or [B, L] with 1 on tokens); the linear-attention
        layers get exactly what the stock forward would derive from it.
    valid: [B, L], 1 on the tokens of each row, right-padded (defaults to attention_mask; None = no padding).
    causal=False: full-attention layers attend bidirectionally within each row (needs enable(body) first);
    causal=True: the stock causal mask (used by the checks).
    inputs_embeds, position_ids: as the multimodal model passes its text decoder (image features in place, and the
        3 x B x L multimodal positions of get_rope_index); by default the embeddings of input_ids and text positions."""
    from transformers.masking_utils import create_causal_mask

    cfg = body.config
    if inputs_embeds is None:
        inputs_embeds = body.embed_tokens(input_ids)
    rows, length = inputs_embeds.shape[:2]
    if position_ids is None:
        position_ids = torch.arange(length, device=inputs_embeds.device).view(1, 1, -1).expand(4, rows, -1)
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids, position_ids = position_ids[0], position_ids[1:]
    else:                                   # multimodal positions: the stock decoder passes no text positions
        text_position_ids = None
    if causal:
        full_mask = create_causal_mask(config=cfg, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                       past_key_values=None, position_ids=text_position_ids)
        extra = {}
    else:
        if any(layer.self_attn.config._attn_implementation != IMPL for layer in _full_layers(body)):
            raise RuntimeError("bidirectional.enable(body) was not called on this model")
        valid = attention_mask if valid is None else valid
        if valid is None:
            lengths = [length] * rows
        else:
            valid = valid.to(device=inputs_embeds.device).bool()
            if tuple(valid.shape) != (rows, length):
                raise ValueError(f"valid mask {tuple(valid.shape)} does not match the input {(rows, length)}")
            n = valid.sum(dim=-1)
            if not bool((valid == (torch.arange(length, device=valid.device)[None] < n[:, None])).all()):
                raise ValueError("bidirectional rows must be right-padded (tokens first, padding last)")
            lengths = n.tolist()
            if min(lengths) < 1:
                raise ValueError("every row needs at least one token")
        full_mask = None
        extra = {"startlux_row_lengths": lengths}
    linear_mask = body._update_linear_attn_mask(attention_mask, None)
    hidden = inputs_embeds
    position_embeddings = body.rotary_emb(hidden, position_ids)
    with _kernel_guard(inputs_embeds.device) if not causal else nullcontext():
        for i, layer in enumerate(body.layers[:cfg.num_hidden_layers]):
            if cfg.layer_types[i] == "linear_attention":
                hidden = layer(hidden, position_embeddings=position_embeddings, attention_mask=linear_mask,
                               position_ids=text_position_ids, past_key_values=None, use_cache=False)
            else:
                hidden = layer(hidden, position_embeddings=position_embeddings, attention_mask=full_mask,
                               position_ids=text_position_ids, past_key_values=None, use_cache=False, **extra)
    return body.norm(hidden)


def _close(a: torch.Tensor, b: torch.Tensor, what: str) -> float:
    diff = float((a.float() - b.float()).abs().max())
    scale = float(a.float().abs().max())
    if not diff <= 1e-3 * (1.0 + scale):
        raise RuntimeError(f"bidirectional check failed: {what} (max diff {diff}, scale {scale})")
    return diff


@torch.no_grad()
def enable(body: Any, tokens: int = 24) -> dict:
    """Route the full-attention layers of `body` through _attention (each gets a shallow config copy naming IMPL; the
    model config stays "sdpa") and check the result on the model itself. Returns a small report."""
    info = check_body(body)
    from transformers import AttentionInterface
    AttentionInterface.register(IMPL, _attention)
    dev = body.embed_tokens.weight.device
    g = torch.Generator(device="cpu").manual_seed(1234)
    ids = torch.randint(0, min(1000, body.config.vocab_size), (2, tokens), generator=g).to(dev)
    am = torch.ones_like(ids)
    am[1, tokens * 2 // 3:] = 0
    refs = [body(input_ids=ids, attention_mask=m, use_cache=False, return_dict=True).last_hidden_state
            for m in (None, am)]
    for layer in _full_layers(body):
        if layer.self_attn.config._attn_implementation != IMPL:
            cfg = copy.copy(layer.self_attn.config)
            cfg._attn_implementation = IMPL
            layer.self_attn.config = cfg
    if body.config._attn_implementation != "sdpa":
        raise RuntimeError("the model config must stay sdpa")
    worst = 0.0
    for ref, m in zip(refs, (None, am)):         # (1) stock forward unchanged, and equal to the causal loop
        worst = max(worst, _close(ref, body(input_ids=ids, attention_mask=m, use_cache=False,
                                             return_dict=True).last_hidden_state, "rerouted stock forward"))
        worst = max(worst, _close(ref, text_forward(body, ids, attention_mask=m, causal=True), "causal loop"))
    n = int(am[1].sum())                         # (2) padding content never reaches a token
    vocab = min(1000, body.config.vocab_size)
    both = text_forward(body, ids, valid=am)
    alt = ids.clone()
    alt[1, n:] = (alt[1, n:] + 7) % vocab
    both2 = text_forward(body, alt, valid=am)
    pad = max(_close(both[0], both2[0], "padding of another row"), _close(both[1, :n], both2[1, :n], "own padding"))
    late = ids.clone()                           # (3) bidirectional: a late token reaches position 0
    late[0, -1] = (late[0, -1] + 1) % vocab
    reach = float((text_forward(body, late, valid=am)[0, 0].float() - both[0, 0].float()).abs().max())
    if not reach > 0:
        raise RuntimeError("bidirectional check failed: a late token does not reach the first position")
    info.update({"impl": IMPL, "causal_loop_max_diff": worst, "padding_invariance_max_diff": pad,
                 "late_token_reach": reach})
    return info
