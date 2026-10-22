"""Low-overhead CUDA synchronization instrumentation."""

from __future__ import annotations

import contextlib
import warnings
from typing import Dict, Iterator

import torch


@contextlib.contextmanager
def count_cuda_syncs() -> Iterator[Dict[str, int]]:
    """Count device-to-host synchronization warnings inside the block."""
    counter = {"n": 0}
    if not torch.cuda.is_available():
        yield counter
        return
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield counter
        finally:
            counter["n"] = sum(
                "call to a synchronizing" in str(item.message).lower()
                for item in caught
            )
            torch.cuda.set_sync_debug_mode(previous)
