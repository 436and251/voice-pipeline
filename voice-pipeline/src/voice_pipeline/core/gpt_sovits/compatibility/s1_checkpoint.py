from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from ..s1 import Text2SemanticDecoder, build_s1_model


def load_s1_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> Text2SemanticDecoder:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("invalid S1 checkpoint: missing config")
    if not isinstance(checkpoint.get("weight"), Mapping):
        raise ValueError("invalid S1 checkpoint: missing weight")

    model = build_s1_model(checkpoint["config"])
    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint["weight"].items()
    }
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


__all__ = ["load_s1_checkpoint"]
