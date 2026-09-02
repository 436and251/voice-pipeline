from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class S2TrainConfig:
    preprocess_dir: Path
    output_dir: Path
    base_s2g_path: Path
    base_s2d_path: Path
    target_optimizer_steps: int
    checkpoint_every_steps: int
    device: str = "cuda:0"
    precision: str = "fp16"
    batch_size: int = 2
    num_workers: int = 0
    seed: int = 1234
    learning_rate: float = 1e-4
    text_low_lr_rate: float = 0.4
    betas: tuple[float, float] = (0.8, 0.99)
    eps: float = 1e-9
    lr_decay: float = 0.999875
    segment_size: int = 20480
    c_mel: float = 45.0
    c_kl: float = 1.0

    def validate(self) -> None:
        if not self.preprocess_dir.is_dir():
            raise ValueError(f"preprocess_dir does not exist: {self.preprocess_dir}")
        for name, path, expected in (
            ("base_s2g_path", self.base_s2g_path, "s2Gv2ProPlus.pth"),
            ("base_s2d_path", self.base_s2d_path, "s2Dv2ProPlus.pth"),
        ):
            if path.name != expected:
                raise ValueError(f"{name} must be named {expected}")
            if not path.is_file():
                raise ValueError(f"{name} does not exist: {path}")
        for name, value in (
            ("target_optimizer_steps", self.target_optimizer_steps),
            ("checkpoint_every_steps", self.checkpoint_every_steps),
            ("batch_size", self.batch_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must not be negative")
        if self.precision not in {"fp16", "fp32"}:
            raise ValueError(f"unsupported precision: {self.precision}")
        try:
            device = torch.device(self.device)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"invalid device: {self.device}") from error
        if self.precision == "fp16" and device.type != "cuda":
            raise ValueError("fp16 S2 training requires CUDA")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
