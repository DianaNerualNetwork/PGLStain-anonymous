"""Sampling strategies for stain-level virtual-staining datasets."""

from __future__ import annotations

import math
import random
from typing import Iterator

from torch.utils.data import Sampler


class StainBalancedSampler(Sampler[list[int]]):
    """Yield batches that mix target stains evenly.

    In MIST/IHC4BC-style datasets each stain may have a different number of
    patches; this sampler round-robins across stains so no single stain
    dominates a batch.

    Args:
        target_stains: ``target_stain`` value for each dataset index.
        batch_size: Desired batch size.
        shuffle: Whether to shuffle within each stain queue.
        seed: Base RNG seed; incremented each epoch so orders differ.
        drop_last: Drop the final incomplete batch.
    """

    def __init__(
        self,
        target_stains: list[str],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self.stain_to_indices: dict[str, list[int]] = {}
        for idx, stain in enumerate(target_stains):
            self.stain_to_indices.setdefault(stain, []).append(idx)

        self.stains = sorted(self.stain_to_indices.keys())

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        self.seed += 1

        queues = {s: idxs.copy() for s, idxs in self.stain_to_indices.items()}
        if self.shuffle:
            for q in queues.values():
                rng.shuffle(q)

        out: list[int] = []
        remaining = sum(len(q) for q in queues.values())
        while remaining > 0:
            for s in self.stains:
                if queues[s]:
                    out.append(queues[s].pop())
                    remaining -= 1

        for i in range(0, len(out), self.batch_size):
            batch = out[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        n = sum(len(v) for v in self.stain_to_indices.values())
        return (
            n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
        )


class SimpleSampler(Sampler[list[int]]):
    """Plain batch sampler; mostly useful for validation.

    Args:
        num_samples: Number of samples in the dataset.
        batch_size: Desired batch size.
        drop_last: Drop the final incomplete batch.
        shuffle: Whether to shuffle the sample order; the order differs each
            epoch so training does not cycle through a fixed sequence.
        seed: Base RNG seed; incremented each epoch so orders differ.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False,
        seed: int = 0,
    ):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(self.num_samples))
        if self.shuffle:
            rng = random.Random(self.seed)
            self.seed += 1
            rng.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        n = self.num_samples
        return (
            n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
        )


__all__ = ["StainBalancedSampler", "SimpleSampler"]
