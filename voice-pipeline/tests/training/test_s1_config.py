from pathlib import Path

import pytest

from voice_pipeline.training.s1.config import S1TrainConfig


def _config(tmp_path: Path, **changes) -> S1TrainConfig:
    preprocess = tmp_path / "preprocess"
    preprocess.mkdir()
    checkpoint = tmp_path / "s1v3.ckpt"
    checkpoint.touch()
    values = dict(
        preprocess_dir=preprocess,
        output_dir=tmp_path / "output",
        base_s1_path=checkpoint,
        target_optimizer_steps=1,
        checkpoint_every_steps=1,
        device="cpu",
        precision="fp32",
    )
    values.update(changes)
    return S1TrainConfig(**values)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"target_optimizer_steps": 0}, "target_optimizer_steps must be positive"),
        ({"checkpoint_every_steps": 0}, "checkpoint_every_steps must be positive"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"gradient_accumulation": 2}, "gradient_accumulation must be 4"),
        ({"precision": "bf16"}, "unsupported precision"),
        ({"precision": "fp16"}, "fp16 S1 training requires CUDA"),
    ],
)
def test_s1_config_rejects_invalid_values(tmp_path: Path, changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, **changes).validate()


def test_s1_config_accepts_cpu_fp32(tmp_path: Path) -> None:
    _config(tmp_path).validate()
