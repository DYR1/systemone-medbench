"""vLLM plugin: prompt-bidirectional, thought-causal attention for Instant model folders (V1 engine), validated on vLLM
0.19.1 and 0.31.0. It patches classes when vLLM imports them; nothing in the installed vLLM is edited. See
docs/vllm_bidirectional.md.

Which models. A folder whose config.json has "jev_prompt_attention": "bidirectional". Active only when the environment
has STARTLUX_PREFIX_BIDIR=1 AND the folder has that value; a causal folder (or the variable unset) runs stock vLLM.

The mask, in the FULL-ATTENTION layers only (positions inside one request):
    query q may read key k   <=>   k <= q   or   (q < p and k < p)
p = the prompt end of the request, from its prompt token ids:
    * the prompt ends with the empty think block "<think>\\n\\n</think>\\n\\n" (one-pass read-out): p = L, all
      bidirectional
    * otherwise p = the position right after the "<think>\\n" that follows the first "<|im_start|>assistant\\n" header;
      the forced prefix, the thought, "</think>" and the read-out after it are causal
    * SamplingParams(extra_args={"startlux_prompt_end": p}), or "vllm_xargs": {"startlux_prompt_end": p} in an
      OpenAI request, sets p explicitly (0 = causal, L = everything bidirectional)
    * no match: an error (STARTLUX_PREFIX_BIDIR_UNMATCHED=causal|full reads such prompts causally / fully)
The Gated DeltaNet layers, their short convolutions and the KV-cache write are untouched.

How. Every full-attention layer first runs the stock causal FlashAttention call of the step. For each request whose
prompt is prefilled in this step, the rows [start, start + p) are recomputed by one non-causal varlen FlashAttention
call over that request's own prompt keys and values (exactly what the stock code wrote to the KV cache) and written
over the causal rows. Rows at or after p keep the causal result, which is the mask above for q >= p. Later steps
need nothing: the prompt's keys and values in the cache come from the bidirectional prompt, and every later query is
later than every prompt key. Decode steps are untouched.

Model runners. The plan of a step (which batch rows to recompute) is made where the model runner builds the attention
metadata of the step:
    * model runner V1 (vllm.v1.worker.gpu_model_runner): GPUModelRunner._build_attention_metadata. The only runner of
      vLLM 0.19.1; on 0.31.0 it runs with VLLM_USE_V2_MODEL_RUNNER=0.
    * model runner V2 (vllm.v1.worker.gpu.model_runner), the default runner of vLLM 0.31.0: the model state's
      prepare_attn, from the step's InputBatch. The prompt token ids and sampling parameters come from the new requests
      of the scheduler output (add_requests). vLLM's kernel warm-up before serving sends synthetic requests through
      the same path (prompt ids 0, 1, 2, ...); they get no plan (read causally, outputs discarded).
Any other runner class (for example V2 on 0.19.1, where it is experimental) is refused at engine start for a
bidirectional folder: a worker whose runner is not patched raises instead of reading the folder causally.

Requirements, checked when the engine starts and on every step (a violation raises; nothing is read with a wrong
mask): chunked prefill and prefix caching off, max_num_batched_tokens >= max_model_len (every prompt is prefilled
whole, from position 0, in one step; a preempted request is recomputed from position 0, also in one step); attention
backend FLASH_ATTN; no speculative decoding, no context parallelism, no ubatching; KV cache bf16/fp16; a step with a
prompt prefill never replays a FULL CUDA graph (the fix-up runs in the eager attention of piecewise and eager steps);
vLLM 0.19.1 or 0.31.0 (STARTLUX_PREFIX_BIDIR_ALLOW_VERSION=1 overrides after re-validation). TRITON_ATTN is refused
unless vLLM is 0.19.1 and STARTLUX_PREFIX_BIDIR_ALLOW_TRITON=1 (read-out-only jobs): in 0.19.1 its PrefixLM path
corrupts decode rows that share a kernel launch with such a prefill, and from 0.31 its PrefixLM ranges are fixed when
the metadata is built, so the plan could not reach them.

Loading. The package declares the entry point ``vllm.general_plugins: startlux_prefix_bidir``; vLLM calls it in the
engine-core process and in every worker before the worker is built, so installing this package in vLLM's Python
environment and setting STARTLUX_PREFIX_BIDIR=1 is all a launcher needs (if VLLM_PLUGINS is set it must list
startlux_prefix_bidir). In-process alternative: VLLM_ENABLE_V1_MULTIPROCESSING=0, then
``from startlux_decide import vllm_plugin; vllm_plugin.install()``. Check an engine with ``require_active(llm, DIR)``;
the server log prints "prefix-bidirectional prompt attention ON for DIR".

Memory. vLLM sizes its KV cache after a profile run whose requests get no plan, so the fix-up's own memory is not
reserved. A step of many short prompts gathers their rows into one call (a few MB). From STARTLUX_PREFIX_BIDIR_LEAN_ROWS
prompt rows in a step (16,384) the fix-up instead reads views of each request's rows and writes the result straight
into the output rows, one call per request, so that it needs no copy of the step's queries, keys, values or outputs
(about 5 GB per layer call for a 262,144-token prompt of the 4B, which ran a full GPU out of memory). Both paths
compute the same attention for every row.

Debug: STARTLUX_PREFIX_BIDIR_CHECK=1 recomputes the attention of up to STARTLUX_PREFIX_BIDIR_CHECK_REQS (2) requests per
layer call in fp32 with the explicit masks and raises when the fix-up is further from its fp32 reference than twice the
stock causal kernel is from its own (see _check).

The variable names and the extra-argument key of the earlier internal copy of this plugin (the same names with the
prefix of the config key instead of STARTLUX / startlux) are accepted as aliases.
"""
from __future__ import annotations

import functools
import importlib.abc
import importlib.util
import inspect
import json
import logging
import os
import sys
import threading

from .modelcfg import PROMPT_ATTENTION_KEY, read_prompt_attention

ENV = "STARTLUX_PREFIX_BIDIR"
LEGACY_ENV = PROMPT_ATTENTION_KEY.split("_", 1)[0].upper() + "_PREFIX_BIDIR"
PLUGIN_NAME = "startlux_prefix_bidir"
VLLM_VERSIONS = ("0.19.1", "0.31.0")          # validated (docs/vllm_bidirectional.md)
LEAN_ROWS = 16384                             # prompt rows in a step from which the fix-up works in place (Memory)
TRITON_VERSION = "0.19.1"                     # the only version whose TRITON_ATTN PrefixLM ranges the plan can set
EXTRA_KEY = "startlux_prompt_end"
LEGACY_EXTRA_KEY = PROMPT_ATTENTION_KEY.split("_", 1)[0] + "_prompt_end"
THINK = "<think>"
ONE_PASS_SUFFIX = "<think>\n\n</think>\n\n"
HEADER = "<|im_start|>assistant\n"
ACTIVE_LINE = "prefix-bidirectional prompt attention ON"      # launchers look for this line in the server log
LINEAR_BACKEND_MODULES = ("gdn_attn", "mamba_attn", "mamba1_attn", "mamba2_attn", "short_conv_attn", "linear_attn")
RUNNER_V1 = "vllm.v1.worker.gpu_model_runner"
RUNNER_V2 = "vllm.v1.worker.gpu.model_runner"
WARMUP_V2 = "vllm.v1.worker.gpu.warmup"
WARMUP_FUNCTIONS = ("warmup_kernels", "run_mixed_prefill_decode_warmup")    # vLLM's synthetic-request warm-ups

logger = logging.getLogger("vllm.startlux_prefix_bidir")
_lock = threading.Lock()
STATUS = {"installed": False, "hooked": [], "patched": [], "error": None, "vllm": None, "pid": os.getpid(),
          "runners": 0, "bi_runners": 0, "steps_with_plan": 0, "bi_requests": 0, "bi_rows": 0, "max_p": 0,
          "explicit": 0, "one_pass": 0, "think": 0, "unmatched": 0, "fixup_calls": 0, "triton_ranges": 0,
          "check_calls": 0, "check_max_diff_prompt": 0.0, "check_max_diff_rest": 0.0, "check_min_bidir_effect": None,
          "check_max_stock_err": 0.0, "check_max_ratio": 0.0, "resumed_prefills": 0, "warmup_steps": 0}


def env(suffix: str = "", default: str | None = None) -> str | None:
    """STARTLUX_PREFIX_BIDIR<suffix>, or its alias from the earlier internal copy."""
    return os.environ.get(ENV + suffix, os.environ.get(LEGACY_ENV + suffix, default))


def _say(msg: str) -> None:
    """To the vLLM log and to stderr (the plugin can run before vLLM's logging is set up)."""
    line = f"[startlux prefix-bidir pid={os.getpid()}] {msg}"
    logger.info(line)
    print(line, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ prompt end
class PromptEnd:
    """p of a prompt from its token ids (see the module docstring). __call__(ids) -> (p or None, kind)."""

    def __init__(self, tokenizer):
        def enc(s: str) -> list:
            return list(tokenizer.encode(s, add_special_tokens=False))
        self.think = tokenizer.convert_tokens_to_ids(THINK)
        nl = enc("\n")
        if len(nl) != 1 or self.think is None or self.think < 0:
            raise ValueError("tokenizer: '<think>' or '\\n' is not a single token")
        self.nl = nl[0]
        self.suffix = enc(ONE_PASS_SUFFIX)
        self.header = enc(HEADER)
        if self.suffix[0] != self.think or self.suffix.count(self.think) != 1:
            raise ValueError(f"unexpected one-pass suffix tokens {self.suffix}")

    def __call__(self, ids) -> tuple:
        n, s, h = len(ids), len(self.suffix), len(self.header)
        if n >= s and list(ids[n - s:]) == self.suffix:
            return n, "one_pass"
        for j, t in enumerate(ids):
            if t == self.think and j + 1 < n and ids[j + 1] == self.nl and j >= h and list(ids[j - h:j]) == self.header:
                return j + 2, "think"
        return None, "unmatched"


# ------------------------------------------------------------------------------------------------ runner side
class _RunnerState:
    def __init__(self, runner, kind: str):
        mc = runner.model_config
        self.model = str(mc.model)
        self.mode = read_prompt_attention(self.model) if os.path.isdir(self.model) else "causal"
        hf_mode = getattr(getattr(mc, "hf_config", None), PROMPT_ATTENTION_KEY, None)
        if hf_mode is not None and hf_mode != self.mode:
            raise RuntimeError(f"{PLUGIN_NAME}: {PROMPT_ATTENTION_KEY} differs between {self.model}/config.json "
                               f"({self.mode}) and the loaded hf_config ({hf_mode})")
        self.bi = self.mode == "bidirectional"
        self.kind = kind                                   # "V1" or "V2": the model runner
        self.tokenizer_path = str(mc.tokenizer or mc.model)
        self.device = runner.device
        self._pe = None
        self.logged = 0
        self.check = env("_CHECK") == "1"

    def prompt_end(self) -> PromptEnd:
        if self._pe is None:
            from transformers import AutoTokenizer
            self._pe = PromptEnd(AutoTokenizer.from_pretrained(self.tokenizer_path))
        return self._pe


def _check_runner_config(runner) -> None:
    """The engine settings the prefix mask needs; raises with what to change."""
    sc, cc, pc, mc = runner.scheduler_config, runner.cache_config, runner.parallel_config, runner.model_config
    bad = []
    if sc.enable_chunked_prefill:
        bad.append("enable_chunked_prefill must be False (LLM(enable_chunked_prefill=False) / "
                   "--no-enable-chunked-prefill)")
    if getattr(sc, "long_prefill_token_threshold", 0):
        bad.append("long_prefill_token_threshold must be 0")
    if sc.max_num_batched_tokens < mc.max_model_len:
        bad.append(f"max_num_batched_tokens ({sc.max_num_batched_tokens}) must be >= max_model_len "
                   f"({mc.max_model_len})")
    if cc.enable_prefix_caching:
        bad.append("enable_prefix_caching must be False (--no-enable-prefix-caching)")
    if runner.speculative_config is not None:
        bad.append("speculative decoding is not supported")
    if getattr(pc, "decode_context_parallel_size", 1) != 1 or getattr(pc, "prefill_context_parallel_size", 1) != 1:
        bad.append("context parallelism is not supported")
    if getattr(pc, "use_ubatching", False):
        bad.append("ubatching (DBO) is not supported")
    if mc.is_encoder_decoder:
        bad.append("encoder-decoder models are not supported")
    if str(cc.cache_dtype) not in ("auto", "bfloat16", "float16"):
        bad.append(f"kv cache dtype {cc.cache_dtype} is not supported (bf16/fp16 only)")
    if getattr(cc, "kv_sharing_fast_prefill", False):
        bad.append("kv_sharing_fast_prefill is not supported")
    if bad:
        raise RuntimeError(f"{PLUGIN_NAME}: the model folder reads its prompt bidirectionally but the engine settings "
                           "do not allow a whole-prompt prefill: " + "; ".join(bad))


def _runner_init(runner, kind: str) -> _RunnerState:
    """After the stock __init__ of a model runner (V1 or V2): the runner state, the engine checks, the log line."""
    st = _RunnerState(runner, kind)
    runner._startlux_prefix = st
    with _lock:
        STATUS["runners"] += 1
        STATUS["bi_runners"] += int(st.bi)
    if st.bi:
        if not STATUS["installed"]:
            raise RuntimeError(f"{PLUGIN_NAME}: {st.model} reads its prompt bidirectionally but the patch is not "
                               f"installed ({STATUS['error']})")
        _check_runner_config(runner)
        _say(f"{ACTIVE_LINE} for {st.model} (model runner {kind}, chunked prefill off, prefix caching off, "
             f"max_num_batched_tokens {runner.scheduler_config.max_num_batched_tokens}, check={st.check})")
    else:
        _say(f"model {st.model} is causal ({PROMPT_ATTENTION_KEY}={st.mode}): stock attention")
    return st


def _check_backends(runner) -> None:
    """After the attention backends are chosen: every full-attention layer must run FLASH_ATTN (see the module
    docstring for TRITON_ATTN)."""
    from vllm.v1.kv_cache_interface import AttentionSpec
    names = set()
    for groups in runner.attn_groups:
        for g in groups:
            if isinstance(g.kv_cache_spec, AttentionSpec):
                names.add(g.backend.get_name())
    if not names or not names <= {"FLASH_ATTN", "TRITON_ATTN"}:
        raise RuntimeError(f"{PLUGIN_NAME}: full-attention layers use {sorted(names)}; the prefix mask needs "
                           "FLASH_ATTN")
    if "TRITON_ATTN" in names and STATUS["vllm"] != TRITON_VERSION:
        raise RuntimeError(f"{PLUGIN_NAME}: TRITON_ATTN is not supported on vLLM {STATUS['vllm']} (its PrefixLM "
                           "ranges are fixed when the metadata is built); use FLASH_ATTN (the default)")
    if "TRITON_ATTN" in names and env("_ALLOW_TRITON") != "1":
        raise RuntimeError(f"{PLUGIN_NAME}: TRITON_ATTN is refused: vLLM 0.19.1's PrefixLM path (USE_MM_PREFIX in "
                           "triton_unified_attention) widens every sequence of a launch to max(causal bound, "
                           "seq_len) keys, and decode rows that share a launch with a prefix-mask prefill read "
                           "past their end (generations collapse into '!!!!' loops; read-outs alone were "
                           f"correct). Use FLASH_ATTN (the default), or {ENV}_ALLOW_TRITON=1 for read-out-only "
                           "jobs")
    _say(f"full-attention backend(s): {sorted(names)}")


def _req_prompt_end(st: _RunnerState, state) -> int:
    p = getattr(state, "_startlux_prompt_end", None)
    if p is not None:
        return p
    ids = state.prompt_token_ids
    if ids is None:
        raise RuntimeError(f"{PLUGIN_NAME}: prompt embeddings are not supported (no prompt token ids)")
    n = len(ids)
    sp = state.sampling_params
    extra = (sp.extra_args or {}) if sp is not None else {}
    ex = extra.get(EXTRA_KEY, extra.get(LEGACY_EXTRA_KEY))
    if ex is not None:
        p, kind = int(ex), "explicit"
        if not 0 <= p <= n:
            raise ValueError(f"{PLUGIN_NAME}: request {state.req_id}: {EXTRA_KEY}={p} outside [0, {n}]")
    else:
        p, kind = st.prompt_end()(ids)
        if p is None:
            policy = env("_UNMATCHED", "error")
            if policy == "causal":
                p = 0
            elif policy == "full":
                p = n
            else:
                raise RuntimeError(f"{PLUGIN_NAME}: request {state.req_id} ({n} prompt tokens) ends neither with "
                                   f"{ONE_PASS_SUFFIX!r} nor contains {HEADER + THINK!r} + newline; pass "
                                   f"SamplingParams(extra_args={{'{EXTRA_KEY}': p}}) or set "
                                   f"{ENV}_UNMATCHED=causal|full")
    state._startlux_prompt_end = p
    with _lock:
        STATUS[kind] += 1
        STATUS["bi_requests"] += 1
        STATUS["max_p"] = max(STATUS["max_p"], p)
    if st.logged < 3:
        st.logged += 1
        _say(f"request {state.req_id}: {n} prompt tokens, prompt end p={p} ({kind})")
    return p


class Plan:
    """The prefix fix-up of one step: rows of the flattened batch to recompute, per-request prompt lengths."""
    __slots__ = ("idx", "cu", "maxp", "reqs", "ntok", "rows", "single")

    def __repr__(self):
        return f"Plan(reqs={self.reqs[:4]}..., rows={self.rows}, maxp={self.maxp}, ntok={self.ntok})"


def _plan(st: _RunnerState, items, ntok: int) -> Plan | None:
    """The plan of a step from the requests whose prompt is not yet computed, in batch order:
    items = (batch index, request state, computed tokens, scheduled tokens, first batch row, prompt tokens).
    None when no request reads a prompt in this step; raises when a prompt would be split across steps."""
    import numpy as np
    import torch
    rows, cu, reqs = [], [0], []
    for i, state, c, s, a0, npt in items:
        p = _req_prompt_end(st, state)
        if p <= 1 or c >= p:          # causal-equivalent, or the prompt is already in the cache (read whole earlier)
            continue
        if c != 0 or s < p:
            raise RuntimeError(f"{PLUGIN_NAME}: request {state.req_id}: its prompt (p={p}) would be split across steps "
                               f"(computed {c}, scheduled {s}); chunked prefill and prefix caching must be off")
        if s > npt:                   # the prefill also holds generated tokens: a request resumed after preemption
            with _lock:
                STATUS["resumed_prefills"] += 1
        rows.append(np.arange(a0, a0 + p, dtype=np.int64))
        cu.append(cu[-1] + p)
        reqs.append((i, a0, s, p))
    if not reqs:
        return None
    plan = Plan()
    plan.idx = torch.from_numpy(np.concatenate(rows)).to(st.device, non_blocking=True)
    plan.cu = torch.tensor(cu, dtype=torch.int32).to(st.device, non_blocking=True)
    plan.maxp = max(r[3] for r in reqs)
    plan.reqs = reqs
    plan.ntok = ntok
    plan.rows = cu[-1]
    plan.single = None
    return plan


def _make_plan(runner, num_reqs: int, num_scheduled_tokens) -> Plan | None:
    """Model runner V1: the plan from the persistent input batch and the scheduled token counts of the step."""
    import numpy as np
    ib = runner.input_batch
    req_ids = ib.req_ids
    if len(req_ids) < num_reqs:
        raise RuntimeError(f"{PLUGIN_NAME}: input batch has {len(req_ids)} requests, step has {num_reqs}")
    nc = ib.num_computed_tokens_cpu[:num_reqs]
    npt = ib.num_prompt_tokens[:num_reqs]
    pre = np.flatnonzero(nc < npt)
    if pre.size == 0:
        return None
    sched = np.fromiter((num_scheduled_tokens[req_ids[i]] for i in range(num_reqs)), dtype=np.int64, count=num_reqs)
    qs = np.zeros(num_reqs + 1, dtype=np.int64)
    np.cumsum(sched, out=qs[1:])
    items = [(i, runner.requests[req_ids[i]], int(nc[i]), int(sched[i]), int(qs[i]), int(npt[i]))
             for i in pre.tolist()]
    return _plan(runner._startlux_prefix, items, int(qs[-1]))


class _NewRequest:
    """Model runner V2: what the plan needs of a request (V2 keeps no per-request object with the prompt ids and the
    sampling parameters)."""
    __slots__ = ("req_id", "prompt_token_ids", "sampling_params", "num_prompt", "_startlux_prompt_end")

    def __init__(self, new_req_data):
        self.req_id = new_req_data.req_id
        self.prompt_token_ids = new_req_data.prompt_token_ids
        self.sampling_params = new_req_data.sampling_params
        self.num_prompt = new_req_data.prompt_len
        self._startlux_prompt_end = None


def _make_plan_v2(runner, ib) -> Plan | None:
    """Model runner V2: the plan from the step's InputBatch (requests in batch order, decodes first). None during
    vLLM's own kernel warm-up (see _warmup_window)."""
    if getattr(runner, "_startlux_warmup", False):
        with _lock:
            STATUS["warmup_steps"] += 1
        return None
    import numpy as np
    n = ib.num_reqs
    pre = np.flatnonzero(ib.is_prefilling_np[:n])
    if pre.size == 0:
        return None
    known = runner._startlux_reqs
    items = []
    for i in pre.tolist():
        state = known.get(ib.req_ids[i])
        if state is None:
            raise RuntimeError(f"{PLUGIN_NAME}: request {ib.req_ids[i]} is prefilled in this step but was not seen "
                               "as a new request")
        items.append((i, state, int(ib.num_computed_tokens_np[i]), int(ib.num_scheduled_tokens[i]),
                      int(ib.query_start_loc_np[i]), int(state.num_prompt)))
    return _plan(runner._startlux_prefix, items, int(ib.num_tokens))


def _attach(attn_metadata, plan: Plan) -> None:
    if isinstance(attn_metadata, list):
        raise RuntimeError(f"{PLUGIN_NAME}: ubatching (a list of metadata) is not supported")
    n_full = 0
    for m in {id(m): m for m in attn_metadata.values()}.values():
        cls = type(m).__name__
        mod = type(m).__module__.rsplit(".", 1)[-1]
        if cls == "FlashAttentionMetadata":
            if m.num_actual_tokens < plan.ntok:
                raise RuntimeError(f"{PLUGIN_NAME}: metadata covers {m.num_actual_tokens} tokens, plan {plan.ntok}")
            if (getattr(m, "causal", True) is not True or getattr(m, "use_cascade", False)
                    or getattr(m, "mm_prefix_query_range_tensor", None) is not None
                    or getattr(m, "rswa_prefix_lens", None) is not None):
                raise RuntimeError(f"{PLUGIN_NAME}: the stock attention of this step is not plain causal attention "
                                   "(per-request causal flags, cascade, multimodal PrefixLM ranges or R-SWA)")
            m.startlux_plan = plan
            n_full += 1
        elif cls == "TritonAttentionMetadata":
            if STATUS["vllm"] != TRITON_VERSION:
                raise RuntimeError(f"{PLUGIN_NAME}: TRITON_ATTN is not supported on vLLM {STATUS['vllm']}")
            rng = dict(m.mm_prefix_range or {})
            for (i, _, _, p) in plan.reqs:
                if rng.get(i):
                    raise RuntimeError(f"{PLUGIN_NAME}: request already has PrefixLM ranges (multimodal)")
                rng[i] = [(0, p - 1)]
            m.mm_prefix_range = rng
            n_full += 1
            with _lock:
                STATUS["triton_ranges"] += 1
        elif mod in LINEAR_BACKEND_MODULES:
            continue
        else:
            raise RuntimeError(f"{PLUGIN_NAME}: full-attention backend metadata {type(m).__module__}.{cls} is not "
                               "supported (use the FLASH_ATTN attention backend)")
    if n_full == 0:
        raise RuntimeError(f"{PLUGIN_NAME}: no full-attention metadata in this step")
    with _lock:
        STATUS["steps_with_plan"] += 1
        STATUS["bi_rows"] += plan.rows


def _patch_runner(mod) -> None:
    """Model runner V1 (vllm.v1.worker.gpu_model_runner)."""
    R = mod.GPUModelRunner
    if getattr(R, "_startlux_patched", False):
        return
    init0, build0, backend0 = R.__init__, R._build_attention_metadata, R.initialize_attn_backend
    sig = inspect.signature(build0)

    def __init__(self, *args, **kwargs):
        init0(self, *args, **kwargs)
        _runner_init(self, "V1")

    def initialize_attn_backend(self, kv_cache_config, *args, **kwargs):
        backend0(self, kv_cache_config, *args, **kwargs)
        st = getattr(self, "_startlux_prefix", None)
        if st is None or not st.bi:
            return
        _check_backends(self)

    def _build_attention_metadata(self, *args, **kwargs):
        out = build0(self, *args, **kwargs)
        st = getattr(self, "_startlux_prefix", None)
        if st is None or not st.bi:
            return out
        b = sig.bind(self, *args, **kwargs)
        b.apply_defaults()
        a = b.arguments
        if a["for_cudagraph_capture"] or a["num_scheduled_tokens"] is None or a["max_query_len"] <= 1:
            return out             # capture / dummy runs, and steps without a multi-token prefill
        plan = _make_plan(self, a["num_reqs"], a["num_scheduled_tokens"])
        if plan is not None:
            _attach(out[0], plan)
        return out

    R.__init__ = __init__
    R.initialize_attn_backend = initialize_attn_backend
    R._build_attention_metadata = _build_attention_metadata
    R._startlux_patched = True
    STATUS["patched"].append("GPUModelRunner")


def _wrap_prepare_attn(runner) -> None:
    """Model runner V2: wrap the model state's prepare_attn (it builds the attention metadata of every step from the
    step's InputBatch) so that the plan is attached to the FLASH_ATTN metadata it returns."""
    ms = runner.model_state
    prep0 = ms.prepare_attn
    if getattr(prep0, "_startlux_wrapped", False):
        return

    def prepare_attn(input_batch, cudagraph_mode, *args, **kwargs):
        out = prep0(input_batch, cudagraph_mode, *args, **kwargs)
        if kwargs.get("for_capture") or not input_batch.has_prefill:
            return out             # capture / dummy batches, and steps without a prefill
        plan = _make_plan_v2(runner, input_batch)
        if plan is not None:
            if getattr(cudagraph_mode, "name", cudagraph_mode) == "FULL":     # vllm.config.compilation.CUDAGraphMode
                raise RuntimeError(f"{PLUGIN_NAME}: a step with a prompt prefill was dispatched to a FULL CUDA graph; "
                                   "the fix-up runs only in eager attention (cudagraph_mode FULL_AND_PIECEWISE, the "
                                   "default, runs such steps piecewise)")
            _attach(out, plan)
        return out

    prepare_attn._startlux_wrapped = True
    ms.prepare_attn = prepare_attn


def _warmup_window(fn):
    """Wraps a vLLM warm-up function whose first argument is the model runner. vLLM 0.31 warms its kernels up (before
    the engine serves anything) with synthetic requests sent through the real execute_model path: prompt ids 0, 1,
    2, ..., outputs discarded. Their prompts match no prompt-end rule, so while such a function runs the runner makes
    no plan and these requests are read causally."""
    if getattr(fn, "_startlux_wrapped", False):
        return fn

    @functools.wraps(fn)
    def wrapper(model_runner, *args, **kwargs):
        prev = getattr(model_runner, "_startlux_warmup", False)
        model_runner._startlux_warmup = True
        try:
            return fn(model_runner, *args, **kwargs)
        finally:
            model_runner._startlux_warmup = prev

    wrapper._startlux_wrapped = True
    return wrapper


def _patch_warmup(mod) -> None:
    """vllm.v1.worker.gpu.warmup (vLLM 0.31): the warm-up functions run inside _warmup_window."""
    for name in WARMUP_FUNCTIONS:
        if hasattr(mod, name):
            setattr(mod, name, _warmup_window(getattr(mod, name)))
    STATUS["patched"].append("warmup")


def _patch_runner_v2(mod) -> None:
    """Model runner V2 (vllm.v1.worker.gpu.model_runner), the default runner of vLLM 0.31."""
    R = mod.GPUModelRunner
    if getattr(R, "_startlux_patched", False):
        return
    init0, load0, kv0 = R.__init__, R.load_model, R.initialize_kv_cache
    add0, rm0, exec0 = R.add_requests, R._remove_request, R.execute_model

    def bi(runner) -> bool:
        st = getattr(runner, "_startlux_prefix", None)
        return st is not None and st.bi

    def __init__(self, *args, **kwargs):
        init0(self, *args, **kwargs)
        self._startlux_reqs = {}
        _runner_init(self, "V2")

    def load_model(self, *args, **kwargs):
        out = load0(self, *args, **kwargs)
        if bi(self):
            _wrap_prepare_attn(self)
        return out

    def initialize_kv_cache(self, kv_cache_config, *args, **kwargs):
        out = kv0(self, kv_cache_config, *args, **kwargs)
        if bi(self):
            _check_backends(self)
        return out

    def add_requests(self, scheduler_output):
        out = add0(self, scheduler_output)
        if bi(self):
            for nr in scheduler_output.scheduled_new_reqs:
                self._startlux_reqs[nr.req_id] = _NewRequest(nr)
        return out

    def _remove_request(self, req_id):
        reqs = getattr(self, "_startlux_reqs", None)
        if reqs:
            reqs.pop(req_id, None)
        return rm0(self, req_id)

    def execute_model(self, *args, **kwargs):
        if bi(self) and not getattr(self.model_state.prepare_attn, "_startlux_wrapped", False):
            raise RuntimeError(f"{PLUGIN_NAME}: the model state of {self._startlux_prefix.model} was replaced after "
                               "load_model; its attention metadata would not get the prefix plan")
        return exec0(self, *args, **kwargs)

    R.__init__ = __init__
    R.load_model = load_model
    R.initialize_kv_cache = initialize_kv_cache
    R.add_requests = add_requests
    R._remove_request = _remove_request
    R.execute_model = execute_model
    R._startlux_patched = True
    STATUS["patched"].append("GPUModelRunnerV2")


# ------------------------------------------------------------------------------------------------ attention side
def _fixup(impl, plan: Plan, query, key, value, output) -> None:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
    if key is None or value is None:
        raise RuntimeError(f"{PLUGIN_NAME}: a full-attention layer without its own keys/values (KV sharing)")
    if impl.alibi_slopes is not None or impl.sinks is not None or impl.kv_cache_dtype.startswith("fp8"):
        raise RuntimeError(f"{PLUGIN_NAME}: ALiBi / sinks / fp8 KV are not supported")
    kw = dict(softmax_scale=impl.scale, causal=False, window_size=list(impl.sliding_window),
              softcap=impl.logits_soft_cap, fa_version=impl.vllm_flash_attn_version,
              num_splits=1 if impl.batch_invariant_enabled else 0)
    if plan.rows >= int(env("_LEAN_ROWS", str(LEAN_ROWS))) and output.dtype == query.dtype:
        import torch
        if plan.single is None:         # once per step: [0, p] of every request, on the device
            plan.single = [torch.tensor([0, p], dtype=torch.int32, device=query.device) for _, _, _, p in plan.reqs]
        for (_, a0, _, p), cu in zip(plan.reqs, plan.single):      # rows [a0, a0 + p): contiguous views, in place
            flash_attn_varlen_func(q=query[a0:a0 + p], k=key[a0:a0 + p], v=value[a0:a0 + p], max_seqlen_q=p,
                                   cu_seqlens_q=cu, max_seqlen_k=p, cu_seqlens_k=cu, out=output[a0:a0 + p], **kw)
        with _lock:
            STATUS["fixup_calls"] += 1
            STATUS["lean_fixup_calls"] = STATUS.get("lean_fixup_calls", 0) + 1
        return
    q = query.index_select(0, plan.idx)
    k = key.index_select(0, plan.idx)
    v = value.index_select(0, plan.idx)
    o = flash_attn_varlen_func(q=q, k=k, v=v, max_seqlen_q=plan.maxp, cu_seqlens_q=plan.cu, max_seqlen_k=plan.maxp,
                               cu_seqlens_k=plan.cu, **kw)
    output.index_copy_(0, plan.idx, o.to(output.dtype))
    with _lock:
        STATUS["fixup_calls"] += 1


def _check(impl, plan: Plan, query, key, value, output, stock_rows) -> None:
    """Debug check of one layer call: fp32 attention with the explicit masks for up to N requests of the step.
        err_bi     fix-up result (prompt rows) vs the fp32 prefix-mask reference
        err_stock  stock causal FlashAttention output of the same prompt rows (kept before the fix-up) vs the fp32
                   causal reference: the kernel precision the fix-up is held to
        err_rest   rows >= p (stock result, kept) vs the fp32 prefix-mask reference
        effect     |prefix-mask reference - causal reference| on the prompt rows (the bidirectional part is not void)
    Raises when err_bi > max(2 err_stock, max|ref| / 128) + 1e-3 (one bf16 ulp at the largest output) or err_rest
    likewise, or the effect is 0 for p > 1."""
    import torch
    nreq = int(env("_CHECK_REQS", "2"))
    lim = int(env("_CHECK_TOKENS", "8192"))
    g = impl.num_heads // impl.num_kv_heads
    cu = plan.cu.tolist()
    for j, (_, a0, s, p) in enumerate(plan.reqs[:nreq]):
        if s > lim:
            continue
        q = query[a0:a0 + s].float()
        k = key[a0:a0 + s].float().repeat_interleave(g, dim=1)
        v = value[a0:a0 + s].float().repeat_interleave(g, dim=1)
        got = output[a0:a0 + s].float()
        stock = stock_rows[cu[j]:cu[j + 1]].float()             # rows [0, p) of this request before the fix-up
        pos = torch.arange(s, device=q.device)
        e_bi = e_st = e_rest = eff = scale = 0.0
        for b0 in range(0, s, 256):
            qi = pos[b0:b0 + 256]
            sc = torch.einsum("qhd,khd->hqk", q[b0:b0 + 256], k) * impl.scale
            causal = pos[None, :] <= qi[:, None]
            mask = causal | ((qi[:, None] < p) & (pos[None, :] < p))
            ref = torch.einsum("hqk,khd->qhd", sc.masked_fill(~mask[None], float("-inf")).softmax(-1), v)
            refc = torch.einsum("hqk,khd->qhd", sc.masked_fill(~causal[None], float("-inf")).softmax(-1), v)
            scale = max(scale, float(ref.abs().max()))
            d = (ref - got[b0:b0 + 256]).abs().amax(dim=(1, 2))
            inp = qi < p
            if bool(inp.any()):
                n_in = int(inp.sum())
                e_bi = max(e_bi, float(d[inp].max()))
                e_st = max(e_st, float((refc[:n_in] - stock[b0:b0 + n_in]).abs().max()))
                eff = max(eff, float((ref - refc)[inp].abs().max()))
            if bool((~inp).any()):
                e_rest = max(e_rest, float(d[~inp].max()))
        # bf16 output: an element of magnitude x is rounded by up to 2^(floor(log2 x) - 8); allow one ulp at the largest
        # output magnitude, or twice the stock causal kernel's own error on the same rows, whichever is larger
        lim_err = max(2.0 * e_st, scale * 2.0 ** -7) + 1e-3
        with _lock:
            STATUS["check_calls"] += 1
            STATUS["check_max_diff_prompt"] = max(STATUS["check_max_diff_prompt"], e_bi)
            STATUS["check_max_diff_rest"] = max(STATUS["check_max_diff_rest"], e_rest)
            STATUS["check_max_stock_err"] = max(STATUS["check_max_stock_err"], e_st)
            STATUS["check_max_ratio"] = max(STATUS["check_max_ratio"], e_bi / max(e_st, 1e-6))
            if p > 1:
                m = STATUS["check_min_bidir_effect"]
                STATUS["check_min_bidir_effect"] = eff if m is None else min(m, eff)
        if STATUS["check_calls"] <= 8 or e_bi > lim_err or e_rest > lim_err:
            _say(f"check: rows [{a0}, {a0 + s}) p={p}: |fixup - fp32 prefix ref| {e_bi:.4g}, |stock causal - fp32 "
                 f"causal ref| {e_st:.4g}, |rest - ref| {e_rest:.4g}, bidirectional effect {eff:.4g}, max|ref| "
                 f"{scale:.4g}")
        if e_bi > lim_err or e_rest > lim_err or (p > 1 and eff == 0.0):
            raise RuntimeError(f"{PLUGIN_NAME} check failed: rows [{a0}, {a0 + s}) p={p}: fixup err {e_bi:.4g}, rest "
                               f"err {e_rest:.4g}, stock causal err {e_st:.4g}, effect {eff:.4g}, limit {lim_err:.4g}")


def _patch_flash_attn(mod) -> None:
    Impl = mod.FlashAttentionImpl
    if getattr(Impl, "_startlux_patched", False):
        return
    fwd0 = Impl.forward
    from vllm.v1.attention.backend import AttentionType

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None,
                output_block_scale=None):
        out = fwd0(self, layer, query, key, value, kv_cache, attn_metadata, output=output, output_scale=output_scale,
                   output_block_scale=output_block_scale)
        plan = getattr(attn_metadata, "startlux_plan", None) if attn_metadata is not None else None
        if plan is not None and self.attn_type == AttentionType.DECODER:
            if self.dcp_world_size > 1:
                raise RuntimeError(f"{PLUGIN_NAME}: decode context parallelism is not supported")
            chk = env("_CHECK") == "1"
            stock_rows = output.index_select(0, plan.idx).clone() if chk else None
            _fixup(self, plan, query, key, value, output)
            if chk:
                _check(self, plan, query, key, value, output, stock_rows)
        return out

    Impl.forward = forward
    Impl._startlux_patched = True
    STATUS["patched"].append("FlashAttentionImpl")


def _status(worker) -> dict:
    mr = getattr(worker, "model_runner", None)
    st = getattr(mr, "_startlux_prefix", None)
    with _lock:
        d = json.loads(json.dumps(STATUS))
    d.update({"pid": os.getpid(), "model": st.model if st else None, "mode": st.mode if st else None,
              "runner": f"{type(mr).__module__}.{type(mr).__name__}" if mr is not None else None,
              "active": bool(st is not None and st.bi and STATUS["installed"]),
              "env": env(), "check": env("_CHECK")})
    return d


def _require_patched_runner(worker) -> None:
    """After the worker built its model runner: a bidirectional folder must have a runner this plugin patched. Catches
    a runner class the plugin does not know (vLLM 0.31 switched its default runner; 0.19.1 has an experimental V2), so
    that such a folder is never read causally without notice."""
    model = str(worker.model_config.model)
    if not (os.path.isdir(model) and read_prompt_attention(model) == "bidirectional"):
        return
    mr = getattr(worker, "model_runner", None)
    st = getattr(mr, "_startlux_prefix", None)
    if st is None or not st.bi:
        why = STATUS["error"] or f"this runner is not supported on vLLM {STATUS['vllm']}"
        raise RuntimeError(f"{PLUGIN_NAME}: {model} reads its prompt bidirectionally but its model runner "
                           f"{type(mr).__module__}.{type(mr).__name__} is not patched by this plugin ({why}); "
                           "vLLM would read the folder causally")


def _patch_worker(mod) -> None:
    W = mod.Worker
    if not hasattr(W, "startlux_prefix_status"):
        W.startlux_prefix_status = _status
        STATUS["patched"].append("Worker.startlux_prefix_status")
    if not getattr(W, "_startlux_guarded", False):
        init_device0 = W.init_device

        def init_device(self, *args, **kwargs):
            out = init_device0(self, *args, **kwargs)
            _require_patched_runner(self)
            return out

        W.init_device = init_device
        W._startlux_guarded = True
        STATUS["patched"].append("Worker.init_device")
    if RUNNER_V2 in _HOOKS:                # the worker's own binding (from .gpu.warmup import warmup_kernels)
        for name in WARMUP_FUNCTIONS:
            if callable(getattr(mod, name, None)):
                setattr(mod, name, _warmup_window(getattr(mod, name)))


def _guard_runner(mod) -> None:
    """Version mismatch: no patch, but a bidirectional folder must not be read causally without notice."""
    R = mod.GPUModelRunner
    if getattr(R, "_startlux_guarded", False):
        return
    init0 = R.__init__

    def __init__(self, *args, **kwargs):
        init0(self, *args, **kwargs)
        mc = self.model_config
        if os.path.isdir(str(mc.model)) and read_prompt_attention(str(mc.model)) == "bidirectional":
            raise RuntimeError(f"{PLUGIN_NAME}: {mc.model} reads its prompt bidirectionally but the patch is not "
                               f"installed: {STATUS['error']}")

    R.__init__ = __init__
    R._startlux_guarded = True


# ------------------------------------------------------------------------------------------------ installation
_HOOKS: dict = {}


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Runs a patch function right after a target module is executed (the module stays the original one)."""

    def find_spec(self, name, path=None, target=None):
        if name not in _HOOKS:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        fn = _HOOKS[name]

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec_):
                return loader.create_module(spec_)

            def exec_module(self, module):
                loader.exec_module(module)
                fn(module)

        spec.loader = _Loader()
        return spec


def _hook(name: str, fn) -> None:
    _HOOKS[name] = fn
    if name in sys.modules:
        fn(sys.modules[name])
    STATUS["hooked"].append(name)


def install() -> bool:
    """Install the import hooks (idempotent). Called by the plugin entry point when STARTLUX_PREFIX_BIDIR=1."""
    if STATUS["installed"] or STATUS["hooked"]:
        return STATUS["installed"]
    try:
        from importlib.metadata import version
        v = version("vllm")
    except Exception as e:                  # noqa: BLE001  (report whatever the metadata lookup says)
        v = f"unknown ({e})"
    STATUS["vllm"] = v
    if not any(isinstance(f, _PostImportFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _PostImportFinder())
    if v not in VLLM_VERSIONS and env("_ALLOW_VERSION") != "1":
        STATUS["error"] = (f"written for vLLM {' and '.join(VLLM_VERSIONS)}, found {v} (set {ENV}_ALLOW_VERSION=1 "
                           "after re-validating)")
        _say(f"NOT installed: {STATUS['error']}")
        _hook(RUNNER_V1, _guard_runner)
        _hook("vllm.v1.worker.gpu_worker", _patch_worker)
        return False
    STATUS["installed"] = True
    _hook(RUNNER_V1, _patch_runner)
    if v != "0.19.1":                       # 0.19.1's V2 runner is experimental and refused (see _require_patched_runner)
        _hook(RUNNER_V2, _patch_runner_v2)
        _hook(WARMUP_V2, _patch_warmup)
    _hook("vllm.v1.attention.backends.flash_attn", _patch_flash_attn)
    _hook("vllm.v1.worker.gpu_worker", _patch_worker)
    _say(f"installed for vLLM {v} (hooks: {', '.join(STATUS['hooked'])})")
    return True


def plugin() -> None:
    """vllm.general_plugins entry point: install when STARTLUX_PREFIX_BIDIR=1, otherwise do nothing at all."""
    if env() == "1":
        install()


def status(llm) -> list:
    """Status dict of every worker of a vllm.LLM; raises if the engine does not have the patch."""
    try:
        return llm.collective_rpc("startlux_prefix_status")
    except Exception as e:                  # noqa: BLE001
        raise RuntimeError(f"{PLUGIN_NAME} is not loaded in the engine ({type(e).__name__}: {e}); install this package "
                           f"in vLLM's Python environment and set {ENV}=1") from e


def require_active(llm, model_dir: str) -> list:
    """For launchers: a bidirectional folder must run with the patch active in every worker; returns the statuses."""
    sts = status(llm) if read_prompt_attention(model_dir) == "bidirectional" else []
    for s in sts:
        if not s.get("active"):
            raise RuntimeError(f"{PLUGIN_NAME}: {model_dir} is bidirectional but an engine worker is not patched: {s}")
    return sts
