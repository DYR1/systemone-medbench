"""The documented ways to run a model folder with fewer bits, as commands to print or run. Nothing is downloaded.

* FP8 in vLLM: no separate build; vLLM quantizes the weights when it loads the folder (``--quantization fp8``).
* GGUF for llama.cpp: llama.cpp's convert_hf_to_gguf.py writes a BF16 GGUF, llama-quantize makes Q8_0, Q4_K_M, ...;
  the tokenizer files, config.json and decision_config.json are copied next to it (the llamacpp backend reads them).
* MLX: mlx_vlm.convert (optionally quantized: the text model only, the vision tower stays BF16), then decision_config.json
  and the tokenizer and processor files copied into the output folder.

A GGUF build of an Instant (prompt-bidirectional) folder needs the converter patch of patches/llama.cpp, which
marks the file non-causal. The MLX engine reads Instant folders bidirectionally itself.
Every build must be measured against BF16 before use (docs/quantization.md): small shifts of the first read-out change
which questions think.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

from .modelcfg import read_prompt_attention

FOLDER_FILES = ("config.json", "decision_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                "merges.txt", "chat_template.jinja", "special_tokens_map.json", "generation_config.json",
                "preprocessor_config.json", "video_preprocessor_config.json")


def fp8_advice(model: str) -> str:
    return (f"FP8 needs no separate build: vLLM quantizes the weights to FP8 when it loads the folder.\n"
            f"  startlux-decide launch-vllm --model {model} --gpus 0 --quantization fp8\n"
            "The KV cache stays BF16 (the prompt-bidirectional plugin refuses an FP8 KV cache). Measure it against "
            "BF16 before use:\n"
            "  startlux-decide eval --url URL_BF16 --data gold.jsonl --out bf16.jsonl\n"
            "  startlux-decide eval --url URL_FP8 --data gold.jsonl --out fp8.jsonl\n"
            "  startlux-decide eval --compare bf16.jsonl fp8.jsonl")


def _converter_marks_bidirectional(llama_cpp: str) -> bool:
    """True when the llama.cpp tree has patches/llama.cpp/0002 (the converter writes attention.causal = false for
    prompt-bidirectional folders)."""
    for root in (os.path.join(llama_cpp, "conversion"), llama_cpp):
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if name.endswith(".py"):
                try:
                    with open(os.path.join(root, name), encoding="utf-8") as f:
                        if "jev_prompt_attention" in f.read():
                            return True
                except OSError:
                    pass
    return False


def gguf_plan(model: str, out: str, llama_cpp: str, qtype: str = "Q8_0") -> tuple[list[list[str]], list[str]]:
    """(commands, problems) of a GGUF build: conversion to BF16, then quantization unless qtype is BF16."""
    problems = []
    convert = os.path.join(llama_cpp, "convert_hf_to_gguf.py")
    if not os.path.isfile(convert):
        problems.append(f"{convert} not found: --llama-cpp must point to a llama.cpp checkout")
    elif read_prompt_attention(model) == "bidirectional" and not _converter_marks_bidirectional(llama_cpp):
        problems.append("this is an Instant (prompt-bidirectional) folder: apply patches/llama.cpp/0002-convert-qwen35-"
                        "prompt-bidirectional.patch to the llama.cpp tree, so the GGUF is marked non-causal (a causal "
                        "build answers as a different model; see docs/engines.md)")
    quantize = shutil.which("llama-quantize") or os.path.join(llama_cpp, "build", "bin", "llama-quantize")
    if qtype.upper() != "BF16" and not (os.path.isfile(quantize) and os.access(quantize, os.X_OK)):
        problems.append("llama-quantize not found on PATH or in LLAMA_CPP/build/bin: build llama.cpp first")
    bf16 = os.path.join(out, "model-BF16.gguf")
    cmds = [[sys.executable, convert, model, "--outtype", "bf16", "--outfile", bf16]]
    if qtype.upper() != "BF16":
        cmds.append([quantize, bf16, os.path.join(out, f"model-{qtype}.gguf"), qtype])
    return cmds, problems


def mlx_plan(model: str, out: str, bits: int | None = None) -> tuple[list[list[str]], list[str]]:
    """(commands, problems) of an MLX build, quantized to `bits` when given."""
    import importlib.util
    problems = [] if importlib.util.find_spec("mlx_vlm") else ["mlx-vlm not found: pip install mlx-vlm (Apple Silicon)"]
    cmd = [sys.executable, "-m", "mlx_vlm", "convert", "--hf-path", model, "--mlx-path", out]
    if bits:
        cmd += ["-q", "--q-bits", str(bits), "--q-group-size", "64"]        # affine; the vision tower is not quantized
    return [cmd], problems


def copy_folder_files(model: str, out: str) -> list[str]:
    """Copy the tokenizer and config files a build needs next to its weights; returns the copied names."""
    os.makedirs(out, exist_ok=True)
    copied = []
    for name in FOLDER_FILES:
        src = os.path.join(model, name)
        if os.path.isfile(src) and not os.path.exists(os.path.join(out, name)):
            shutil.copy2(src, os.path.join(out, name))
            copied.append(name)
    return copied


def run(cmds: list[list[str]]) -> None:
    for cmd in cmds:
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
