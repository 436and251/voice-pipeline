from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from voice_pipeline.core.gpt_sovits.compatibility.checkpoints import load_sovits, save_sovits
from voice_pipeline.exporting.checkpoints import export_s1_checkpoint, export_s2_checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _s1_base(path: Path) -> None:
    torch.save(
        {"weight": {"model.base": torch.tensor([1.0])}, "config": {"model": {"vocab_size": 1025}}, "info": "base"},
        path,
    )


def _s2_base(path: Path) -> None:
    save_sovits(
        path,
        {"weight": {"base": torch.tensor([1.0])}, "config": {"model": {"semantic_frame_rate": "25hz"}}, "info": "base"},
    )


def test_exports_internal_s1_as_official_fp16_envelope(tmp_path: Path):
    base, source, destination = tmp_path / "base.ckpt", tmp_path / "step.pt", tmp_path / "s1.ckpt"
    _s1_base(base)
    torch.save(
        {
            "format_version": 1,
            "profile": "v2ProPlus",
            "optimizer_step": 12,
            "model": {"layer.weight": torch.tensor([1.25]), "counter": torch.tensor(3)},
        },
        source,
    )

    metadata = export_s1_checkpoint(source, base, destination)
    exported = torch.load(destination, map_location="cpu", weights_only=False)

    assert set(exported) == {"weight", "config", "info"}
    assert set(exported["weight"]) == {"model.layer.weight", "model.counter"}
    assert exported["weight"]["model.layer.weight"].dtype == torch.float16
    assert exported["weight"]["model.counter"].dtype == torch.int64
    assert exported["config"] == torch.load(base, map_location="cpu", weights_only=False)["config"]
    assert metadata == {
        "source_sha256": _sha256(source),
        "exported_sha256": _sha256(destination),
        "source_kind": "training_checkpoint",
        "optimizer_step": 12,
    }


def test_exports_base_s1_without_training_state(tmp_path: Path):
    base, destination = tmp_path / "base.ckpt", tmp_path / "s1.ckpt"
    _s1_base(base)
    metadata = export_s1_checkpoint(base, base, destination)
    exported = torch.load(destination, map_location="cpu", weights_only=False)
    assert exported["weight"]["model.base"].dtype == torch.float16
    assert metadata["source_kind"] == "official_base"
    assert metadata["optimizer_step"] is None


def test_exports_internal_s2_generator_only_with_official_codec(tmp_path: Path):
    base, source, destination = tmp_path / "base.pth", tmp_path / "step.pt", tmp_path / "s2.pth"
    _s2_base(base)
    torch.save(
        {
            "format_version": 1,
            "profile": "v2ProPlus",
            "global_step": 23,
            "net_g": {
                "decoder.weight": torch.tensor([2.5]),
                "enc_q.secret": torch.tensor([9.0]),
                "counter": torch.tensor(4),
            },
            "net_d": {"weight": torch.tensor([8.0])},
        },
        source,
    )

    metadata = export_s2_checkpoint(source, base, destination)
    raw = destination.read_bytes()
    exported = load_sovits(destination)

    assert raw[:2] == b"06" and raw[2:4] != b"PK"
    assert set(exported) == {"weight", "config", "info"}
    assert set(exported["weight"]) == {"decoder.weight", "counter"}
    assert exported["weight"]["decoder.weight"].dtype == torch.float16
    assert exported["weight"]["counter"].dtype == torch.int64
    assert metadata["source_kind"] == "training_checkpoint"
    assert metadata["optimizer_step"] == 23


def test_exports_base_s2_in_normalized_inference_form(tmp_path: Path):
    base, destination = tmp_path / "base.pth", tmp_path / "s2.pth"
    _s2_base(base)
    metadata = export_s2_checkpoint(base, base, destination)
    assert load_sovits(destination)["weight"]["base"].dtype == torch.float16
    assert metadata["source_kind"] == "official_base"
    assert metadata["optimizer_step"] is None


@pytest.mark.parametrize("stage", ["s1", "s2"])
def test_rejects_wrong_profile_and_nonfinite_weights_without_output(tmp_path: Path, stage: str):
    base = tmp_path / f"base.{stage}"
    source = tmp_path / "step.pt"
    destination = tmp_path / "output.pt"
    if stage == "s1":
        _s1_base(base)
        payload = {"format_version": 1, "profile": "v2", "optimizer_step": 1, "model": {"x": torch.tensor(float("nan"))}}
        export = export_s1_checkpoint
    else:
        _s2_base(base)
        payload = {"format_version": 1, "profile": "v2", "global_step": 1, "net_g": {"x": torch.tensor(float("nan"))}}
        export = export_s2_checkpoint
    torch.save(payload, source)

    with pytest.raises(ValueError, match="profile"):
        export(source, base, destination)
    assert not destination.exists()

    payload["profile"] = "v2ProPlus"
    torch.save(payload, source)
    with pytest.raises(ValueError, match="finite"):
        export(source, base, destination)
    assert not destination.exists()


def test_rejects_discriminator_only_and_destination_alias(tmp_path: Path):
    base, source = tmp_path / "base.pth", tmp_path / "step.pt"
    _s2_base(base)
    torch.save({"format_version": 1, "profile": "v2ProPlus", "global_step": 1, "net_d": {"x": torch.ones(1)}}, source)
    with pytest.raises(ValueError, match="generator"):
        export_s2_checkpoint(source, base, tmp_path / "output.pth")
    with pytest.raises(ValueError, match="must differ"):
        export_s2_checkpoint(base, base, base)
