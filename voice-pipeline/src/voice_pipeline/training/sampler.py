from __future__ import annotations

from collections.abc import Sized

import torch
from torch.utils.data import Sampler


class DeterministicEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


__all__ = ["DeterministicEpochSampler"]
