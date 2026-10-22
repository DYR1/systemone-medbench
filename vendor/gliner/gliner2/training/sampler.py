"""Deterministic length-grouped samplers for dynamically padded training."""

from __future__ import annotations

import math
from typing import Iterator, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class LengthGroupedSampler(Sampler[int]):
    """Shuffle, locally sort by length, then shuffle batches.

    Every dataset index is emitted exactly once per epoch. ``set_epoch`` makes
    the order deterministic for a given ``seed`` and epoch without correlating
    batch order across epochs.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        window_batches: int = 50,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if window_batches <= 0:
            raise ValueError("window_batches must be > 0")
        self.lengths = tuple(int(length) for length in lengths)
        self.batch_size = int(batch_size)
        self.window_batches = int(window_batches)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.lengths), generator=generator).tolist()
        window = self.batch_size * self.window_batches
        batches: list[list[int]] = []
        for start in range(0, len(order), window):
            group = order[start:start + window]
            group.sort(key=self.lengths.__getitem__)
            batches.extend(
                group[offset:offset + self.batch_size]
                for offset in range(0, len(group), self.batch_size)
            )
        if batches:
            full = [batch for batch in batches if len(batch) == self.batch_size]
            partial = [batch for batch in batches if len(batch) != self.batch_size]
            batch_order = torch.randperm(len(full), generator=generator).tolist()
            # A partial batch must remain last because DataLoader receives a
            # flat index stream and reconstructs batch boundaries itself.
            batches = [full[index] for index in batch_order] + partial
        return batches

    def __iter__(self) -> Iterator[int]:
        return iter(index for batch in self._batches() for index in batch)

    def __len__(self) -> int:
        return len(self.lengths)


class DistributedLengthGroupedSampler(LengthGroupedSampler):
    """Length grouping with disjoint, deterministic batch-level rank shards.

    Unlike ``DistributedSampler`` this sampler never pads by repeating indices.
    Whole batches are assigned round-robin to ranks; a small number of trailing
    batches may be omitted so every rank executes the same number of steps.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        window_batches: int = 50,
        seed: int = 0,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("distributed process group is not initialized")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("distributed process group is not initialized")
            rank = dist.get_rank()
        if num_replicas <= 0:
            raise ValueError("num_replicas must be > 0")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        super().__init__(
            lengths,
            batch_size,
            window_batches=window_batches,
            seed=seed,
        )
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def _rank_batches(self) -> list[list[int]]:
        # Never pad or duplicate the final partial batch. Dropping it also
        # keeps sample counts and optimizer-step counts equal across ranks.
        batches = [
            batch for batch in self._batches()
            if len(batch) == self.batch_size
        ]
        usable = len(batches) - (len(batches) % self.num_replicas)
        return batches[:usable][self.rank::self.num_replicas]

    def __iter__(self) -> Iterator[int]:
        return iter(index for batch in self._rank_batches() for index in batch)

    def __len__(self) -> int:
        total_batches = math.ceil(len(self.lengths) / self.batch_size)
        rank_batches = total_batches // self.num_replicas
        # All assigned batches are full except possibly the final global batch.
        # Compute exactly because DataLoader uses this for its own length.
        return sum(len(batch) for batch in self._rank_batches())


__all__ = ["LengthGroupedSampler", "DistributedLengthGroupedSampler"]
