from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from ..s2_v2proplus import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
    build_s2_discriminator,
    build_s2_generator,
)


def _load_checkpoint(path: str | Path) -> Mapping[str, object]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("invalid S2 checkpoint: missing config")
    if not isinstance(checkpoint.get("weight"), Mapping):
        raise ValueError("invalid S2 checkpoint: missing weight")
    return checkpoint


def load_s2_generator(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> SynthesizerTrn:
    checkpoint = _load_checkpoint(path)
    model = build_s2_generator(checkpoint["config"])
    model.load_state_dict(checkpoint["weight"], strict=True)
    return model.to(device)


def load_s2_discriminator(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> MultiPeriodDiscriminator:
    checkpoint = _load_checkpoint(path)
    use_spectral_norm = bool(checkpoint["config"]["model"]["use_spectral_norm"])
    model = build_s2_discriminator(use_spectral_norm=use_spectral_norm)
    model.load_state_dict(checkpoint["weight"], strict=True)
    return model.to(device)


__all__ = ["load_s2_discriminator", "load_s2_generator"]
