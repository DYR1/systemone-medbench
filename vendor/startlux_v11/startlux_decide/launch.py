"""Start vLLM OpenAI-compatible servers for a model folder, one per GPU, with the flags of the release runs.

Why each flag (docs/engines.md has the details):

* ``--logprobs-mode processed_logprobs --max-logprobs 26``: the read-out asks for the log-probabilities of up to 26
  allowed letters; processed log-probabilities are taken after the allowed-token mask, so all of them come back.
* ``--generation-config vllm``: otherwise the folder's generation_config (top_k, top_p) is applied to the read-out
  and cuts letters off.
* ``--max-model-len``: on vLLM 0.31 or later the folder's native context (``max_position_embeddings``, 262,144 tokens
  in these folders) for causal folders, and 131,072 for Instant folders, unless set lower; on 0.19.1, and when the
  folder does not say, 65,536 (the release setting). An Instant prompt is prefilled in one step, and on vLLM 0.31 a
  step of a 196,000-token prompt (27B) or a 256,000-token prompt (4B) crashed the server (an illegal memory access;
  128,000 tokens ran at both sizes). A longer ``--max-model-len`` can be set, at that risk.
* Images (vLLM 0.31 or later, folders with a vision tower, unless switched off): ``--limit-mm-per-prompt
  '{"image":N,"video":0}'`` (N = most images per request, 8 by default), ``--mm-processor-kwargs`` with the resize
  bounds (at most 1,048,576 pixels per image) and ``--trust-request-chat-template`` (image calls carry the client's
  pass-through template). Otherwise ``--limit-mm-per-prompt '{"image":0,"video":0}'``: text only, and no memory for
  the vision tower's inputs. Instant servers with images also get ``--skip-mm-profiling``: vLLM profiles the image
  encoder with as many full-size images as one step holds (256 when a step holds 262,144 tokens); with that profile
  the 4B Instant server did not start (an illegal memory access in the profile run), without it the server started
  and read image requests.

Prefix caching, chunked prefill and speculative decoding depend on the vLLM version and the folder:

* vLLM 0.19.1 (and any folder on a vLLM older than 0.31, or whose version cannot be read):
  ``--no-enable-prefix-caching --no-enable-chunked-prefill`` and ``--max-num-batched-tokens`` = ``--max-model-len``.
  On 0.19.1 both change answers of these hybrid models silently (prefix caching moved 2.6% of ordinary and 16% of
  wide-list read-outs by more than 0.05 in our tests); a whole prompt fits in one step.
* AutoThink (causal) folders on vLLM 0.31 or later: ``--enable-prefix-caching --enable-chunked-prefill
  --max-num-batched-tokens 8192``. On 0.31 prefix caching works on these hybrid models only together with chunked
  prefill; with both on, the 27B AutoThink model gave the same answer as on 0.19.1 for 99.88% of 823 questions. When the
  folder holds MTP weights (mtp.* tensors and ``mtp_num_hidden_layers`` in config.json), the servers also get MTP
  speculative decoding with 3 draft tokens (``--speculative-config '{"method":"mtp","num_speculative_tokens":3}'``):
  the thoughts keep their length distribution and are generated faster. ``mtp=False`` switches it off.
* Instant (prompt-bidirectional) folders, on every version: both off and no speculative decoding, because the plugin
  needs every prompt prefilled whole in one step (``--max-num-batched-tokens`` = ``--max-model-len``: one step may
  hold a prompt of the full context). The servers get STARTLUX_PREFIX_BIDIR=1, and the launcher refuses to go on
  unless every server log shows that the plugin is active.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .backends.vllm import STREAM_VERSION, parse_version
from .images import MAX_PIXELS, MIN_PIXELS
from .modelcfg import native_context, read_config, read_prompt_attention, vision_settings
from .vllm_plugin import ACTIVE_LINE, ENV, PLUGIN_NAME

MAX_MODEL_LEN = 65536               # when the folder does not give its native context
GPU_MEMORY_UTILIZATION = 0.88
INSTANT_MAX_MODEL_LEN = 131072      # default context of Instant servers on vLLM 0.31 or later (module docstring)
CACHED_BATCHED_TOKENS = 8192        # --max-num-batched-tokens with chunked prefill (vLLM 0.31 or later)
MTP_DRAFT_TOKENS = 3                # MTP speculative decoding: draft tokens per step
MAX_IMAGES = 8                      # images per request a server takes (launch --max-images)


def server_command(python: str, model: str, port: int, host: str = "127.0.0.1", served_name: str = "m",
                   max_model_len: int = MAX_MODEL_LEN, gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
                   quantization: str | None = None, prefix_caching: bool = False, mtp_tokens: int = 0,
                   images: int = 0, max_pixels: int = MAX_PIXELS) -> list[str]:
    """The command line of one vLLM server with the release flags. `prefix_caching`: prefix caching and chunked
    prefill on (AutoThink folders on vLLM 0.31 or later), else both off; `mtp_tokens` > 0: MTP speculative
    decoding with that many draft tokens; `images` > 0: that many images per request, resized to at most
    `max_pixels` pixels (else text only)."""
    cmd = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--served-model-name", served_name,
           "--host", host, "--port", str(port), "--max-model-len", str(max_model_len),
           "--gpu-memory-utilization", str(gpu_memory_utilization), "--logprobs-mode", "processed_logprobs",
           "--max-logprobs", "26", "--generation-config", "vllm"]
    if prefix_caching:
        cmd += ["--enable-prefix-caching", "--enable-chunked-prefill", "--max-num-batched-tokens",
                str(CACHED_BATCHED_TOKENS)]
    else:
        cmd += ["--no-enable-prefix-caching", "--no-enable-chunked-prefill", "--max-num-batched-tokens",
                str(max_model_len)]
    cmd += ["--limit-mm-per-prompt", json.dumps({"image": images, "video": 0}, separators=(",", ":"))]
    if images:
        size = {"size": {"longest_edge": max_pixels, "shortest_edge": MIN_PIXELS}}
        cmd += ["--mm-processor-kwargs", json.dumps(size, separators=(",", ":")), "--trust-request-chat-template"]
        if not prefix_caching:              # a step may hold a whole prompt: see the module docstring
            cmd += ["--skip-mm-profiling"]
    cmd += ["--disable-uvicorn-access-log"]
    if mtp_tokens:
        cmd += ["--speculative-config",
                json.dumps({"method": "mtp", "num_speculative_tokens": mtp_tokens}, separators=(",", ":"))]
    if quantization:
        cmd += ["--quantization", quantization]
    return cmd


def vllm_version(python: str) -> str | None:
    """The vLLM version installed for `python` (read from its package metadata; vLLM is not imported). None when
    vLLM is not installed there."""
    code = "import importlib.metadata as m; print(m.version('vllm'))"
    try:
        r = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (r.stdout.strip() or None) if r.returncode == 0 else None


def has_mtp_weights(model: str) -> bool:
    """True when the folder declares MTP layers (``mtp_num_hidden_layers`` >= 1 in config.json, top level or
    text_config) and holds their weights (``mtp.*`` tensors in model.safetensors.index.json, or in model.safetensors
    for a single-file folder)."""
    try:
        with open(os.path.join(model, "config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return False
    layers = (cfg.get("text_config") or {}).get("mtp_num_hidden_layers", cfg.get("mtp_num_hidden_layers")) or 0
    if not isinstance(layers, int) or layers < 1:
        return False
    names: list = []
    index = os.path.join(model, "model.safetensors.index.json")
    single = os.path.join(model, "model.safetensors")
    try:
        if os.path.isfile(index):
            with open(index) as f:
                names = list((json.load(f).get("weight_map") or {}))
        elif os.path.isfile(single):
            with open(single, "rb") as f:
                size = int.from_bytes(f.read(8), "little")
                names = list(json.loads(f.read(size)))
    except (OSError, ValueError):
        return False
    return any(n.startswith("mtp.") for n in names)


def launch_variables(gpu: str, bidirectional: bool) -> dict:
    """The variables the launcher sets for one server: its GPU, no usage statistics, the plugin switch."""
    out = {"CUDA_VISIBLE_DEVICES": gpu, "VLLM_NO_USAGE_STATS": "1"}
    if bidirectional:
        out[ENV] = "1"
    return out


def server_env(gpu: str, bidirectional: bool, base: dict | None = None) -> dict:
    """Environment of one server: `base` (default: this process's) with the launch variables."""
    env = dict(os.environ if base is None else base)
    env.update(launch_variables(gpu, bidirectional))
    if bidirectional:
        plugins = env.get("VLLM_PLUGINS")
        if plugins is not None and PLUGIN_NAME not in [p.strip() for p in plugins.split(",")]:
            raise RuntimeError(f"VLLM_PLUGINS is set to {plugins!r} without {PLUGIN_NAME}; vLLM would not load the "
                               "prompt-bidirectional plugin")
    return env


def plugin_visible(python: str) -> bool:
    """True when the Python that runs vLLM sees this package's vLLM plugin entry point."""
    group = "vllm.general_plugins"
    code = ("import importlib.metadata as m, sys; eps = m.entry_points(); "
            f"g = eps.select(group={group!r}) if hasattr(eps, 'select') else eps.get({group!r}, []); "
            f"sys.exit(0 if any(e.name == {PLUGIN_NAME!r} for e in g) else 1)")
    return subprocess.run([python, "-c", code], capture_output=True).returncode == 0


def plugin_problem(log_text: str) -> str | None:
    """None when a server log shows the plugin active, else what went wrong."""
    if ACTIVE_LINE in log_text:
        return None
    for line in log_text.splitlines():
        if "startlux prefix-bidir" in line and "NOT installed" in line:
            return line.strip()
    return (f"the log has no '{ACTIVE_LINE}' line: vLLM did not load the plugin (install this package in the vLLM "
            f"environment and keep {ENV}=1)")


@dataclass
class VLLMServers:
    """N vLLM servers for one folder (one per GPU), their logs and their lifetime."""
    model: str
    gpus: list
    base_port: int = 8001
    host: str = "127.0.0.1"
    log_dir: str = "vllm_logs"
    python: str = sys.executable
    served_name: str = "m"
    max_model_len: int | None = None        # default: see the module docstring
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION
    quantization: str | None = None
    mtp: bool = True                        # MTP speculative decoding when the folder and the vLLM version allow it
    vllm_version: str | None = None         # default: read from `python`
    max_images: int = MAX_IMAGES            # images per request on vLLM 0.31 or later (0: text only)
    max_pixels: int = MAX_PIXELS
    procs: list = field(default_factory=list)

    def __post_init__(self):
        if self.max_model_len is None:
            native = native_context(read_config(self.model)) if self.modern else None
            if native and self.bidirectional:
                native = min(native, INSTANT_MAX_MODEL_LEN)
            self.max_model_len = native or MAX_MODEL_LEN

    @property
    def modern(self) -> bool:
        """vLLM 0.31 or later."""
        v = parse_version(self.version)
        return v is not None and v >= STREAM_VERSION

    @property
    def bidirectional(self) -> bool:
        return read_prompt_attention(self.model) == "bidirectional"

    @property
    def version(self) -> str | None:
        """The vLLM version of `python` (read once)."""
        if self.vllm_version is None:
            self.vllm_version = vllm_version(self.python) or ""
        return self.vllm_version or None

    @property
    def prefix_caching(self) -> bool:
        """Prefix caching and chunked prefill on: an AutoThink (causal) folder on vLLM 0.31 or later."""
        return self.modern and not self.bidirectional

    @property
    def images(self) -> int:
        """Images per request the servers take: on vLLM 0.31 or later, for a folder with a vision tower."""
        return self.max_images if self.modern and vision_settings(read_config(self.model))[0] else 0

    @property
    def mtp_tokens(self) -> int:
        """MTP draft tokens per step: 3 when prefix caching is on, the folder holds MTP weights and `mtp` is set."""
        return MTP_DRAFT_TOKENS if self.mtp and self.prefix_caching and has_mtp_weights(self.model) else 0

    def settings(self) -> str:
        """One line on the version-dependent settings, for logs."""
        tail = f"; context {self.max_model_len} tokens; " + (
            f"up to {self.images} images per request" if self.images else "text only")
        if not self.prefix_caching:
            return (f"vLLM {self.version or '(version unknown)'}: prefix caching and chunked prefill off"
                    + (" (Instant folder)" if self.bidirectional else "") + tail)
        mtp = f"MTP speculative decoding with {self.mtp_tokens} draft tokens" if self.mtp_tokens else (
            "no MTP (switched off)" if not self.mtp else "no MTP (the folder holds no MTP weights)")
        return f"vLLM {self.version}: prefix caching and chunked prefill on, {mtp}" + tail

    @property
    def urls(self) -> list[str]:
        return [f"http://{self.host}:{self.base_port + i}" for i in range(len(self.gpus))]

    def log_path(self, i: int) -> str:
        return os.path.join(self.log_dir, f"vllm_{i}_gpu{self.gpus[i]}.log")

    def plan(self) -> list[tuple[list[str], dict]]:
        """(command, environment) of every server."""
        caching, mtp, images = self.prefix_caching, self.mtp_tokens, self.images
        return [(server_command(self.python, self.model, self.base_port + i, self.host, self.served_name,
                                self.max_model_len, self.gpu_memory_utilization, self.quantization, caching, mtp,
                                images, self.max_pixels),
                 server_env(str(g), self.bidirectional)) for i, g in enumerate(self.gpus)]

    def start(self) -> None:
        if self.bidirectional and not plugin_visible(self.python):
            raise RuntimeError(f"{self.model} is an Instant (prompt-bidirectional) folder, and {self.python} does not "
                               f"see the {PLUGIN_NAME} vLLM plugin: install this package in the Python that runs vLLM "
                               "(pip install -e . in that environment)")
        os.makedirs(self.log_dir, exist_ok=True)
        for i, (cmd, env) in enumerate(self.plan()):
            log = open(self.log_path(i), "w")
            self.procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True))
            log.close()

    def wait_ready(self, timeout: float = 1800.0) -> None:
        """Wait until every server answers /health; raises when one exits or the time is up."""
        deadline = time.time() + timeout
        pending = set(range(len(self.procs)))
        while pending:
            for i in sorted(pending):
                if self.procs[i].poll() is not None:
                    raise RuntimeError(f"vLLM server {i} exited with code {self.procs[i].returncode}; last lines of "
                                       f"{self.log_path(i)}:\n{self.tail(i)}")
                try:
                    with urllib.request.urlopen(self.urls[i] + "/health", timeout=2) as r:
                        if r.status == 200:
                            pending.discard(i)
                except (urllib.error.URLError, OSError):
                    pass
            if pending and time.time() > deadline:
                raise RuntimeError(f"vLLM servers {sorted(pending)} not ready after {timeout:.0f} s")
            if pending:
                time.sleep(5)

    def check_plugin(self) -> None:
        """For an Instant folder: every server log must show the plugin active."""
        if not self.bidirectional:
            return
        for i in range(len(self.procs)):
            with open(self.log_path(i), errors="replace") as f:
                problem = plugin_problem(f.read())
            if problem:
                raise RuntimeError(f"vLLM server {i} ({self.log_path(i)}) would read the Instant folder causally: "
                                   f"{problem}")

    def tail(self, i: int, lines: int = 30) -> str:
        with open(self.log_path(i), errors="replace") as f:
            return "".join(f.readlines()[-lines:])

    def stop(self, grace: float = 60.0) -> None:
        """Terminate every server's process group (the API process and the engine processes it started), and kill
        what is left after `grace` seconds."""
        for p in self.procs:
            _signal_group(p.pid, signal.SIGTERM)
        deadline = time.time() + grace
        for p in self.procs:
            try:
                p.wait(max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                pass
            while _signal_group(p.pid, 0) and time.time() < deadline:
                time.sleep(0.5)
            _signal_group(p.pid, signal.SIGKILL)
            p.wait()
        self.procs.clear()

    def up(self, timeout: float = 1800.0) -> None:
        """Start, wait until ready and check the plugin; stops every server when any of that fails."""
        try:
            self.start()
            self.wait_ready(timeout)
            self.check_plugin()
        except BaseException:
            self.stop()
            raise


def _signal_group(pgid: int, sig: int) -> bool:
    """Send `sig` to a process group; False when the group has no process left (signal 0 only checks)."""
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False
