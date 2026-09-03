from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from .checkpoints import load_sovits
from ..s2_v2proplus import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
    build_s2_discriminator,
    build_s2_generator,
)


def _load_checkpoint(path: str | Path) -> Mapping[str, object]:
    checkpoint = load_sovits(path)
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
    try:
        incompatible = model.load_state_dict(checkpoint["weight"], strict=False)
    except RuntimeError as error:
        raise ValueError(f"invalid S2 inference weights: {error}") from error
    missing = [key for key in incompatible.missing_keys if not key.startswith("enc_q.")]
    if missing or incompatible.unexpected_keys:
        raise ValueError(
            "invalid S2 inference weights: "
            f"missing={missing}, unexpected={incompatible.unexpected_keys}"
        )
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
