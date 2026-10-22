"""Classification of accelerator errors that poison the current process."""

from __future__ import annotations

import torch


_FATAL_DEVICE_MARKERS = (
    "cuda error",
    "device-side assert",
    "illegal memory access",
    "an illegal memory access",
    "cudnn_status",
    "hip error",
    "mps backend",
)


def is_fatal_device_error(error: BaseException) -> bool:
    """Return whether training must terminate instead of skipping the batch."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return False
    accelerator_error = getattr(torch, "AcceleratorError", None)
    if accelerator_error is not None and isinstance(error, accelerator_error):
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return any(marker in message for marker in _FATAL_DEVICE_MARKERS)


__all__ = ["is_fatal_device_error"]
