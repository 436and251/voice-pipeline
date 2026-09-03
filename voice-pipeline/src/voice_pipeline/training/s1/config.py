from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class S1TrainConfig:
    preprocess_dir: Path
    output_dir: Path
    base_s1_path: Path
    target_optimizer_steps: int
    checkpoint_every_steps: int
    device: str = "cuda:0"
    precision: str = "fp16"
    batch_size: int = 2
    num_workers: int = 0
    seed: int = 1234
    gradient_accumulation: int = 4
    max_sec: int = 57
    hz: int = 25
    min_ps_ratio: float = 3.0
    max_ps_ratio: float = 25.0

    def validate(self) -> None:
        if not self.preprocess_dir.is_dir():
            raise ValueError(f"preprocess_dir does not exist: {self.preprocess_dir}")
        if self.base_s1_path.name != "s1v3.ckpt":
            raise ValueError("base_s1_path must be named s1v3.ckpt")
        if not self.base_s1_path.is_file():
            raise ValueError(f"base_s1_path does not exist: {self.base_s1_path}")
        for name, value in (
            ("target_optimizer_steps", self.target_optimizer_steps),
            ("checkpoint_every_steps", self.checkpoint_every_steps),
            ("batch_size", self.batch_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.gradient_accumulation != 4:
            raise ValueError("gradient_accumulation must be 4")
        if self.num_workers < 0:
            raise ValueError("num_workers must not be negative")
        if self.precision not in {"fp16", "fp32"}:
            raise ValueError(f"unsupported precision: {self.precision}")
        try:
            device = torch.device(self.device)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"invalid device: {self.device}") from error
        if self.precision == "fp16" and device.type != "cuda":
            raise ValueError("fp16 S1 training requires CUDA")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")


__all__ = ["S1TrainConfig"]
