from pathlib import Path

import pytest

from voice_pipeline.training.s2.config import S2TrainConfig


def _config(tmp_path: Path, **changes) -> S2TrainConfig:
    preprocess = tmp_path / "preprocess"
    preprocess.mkdir(exist_ok=True)
    generator = tmp_path / "s2Gv2ProPlus.pth"
    discriminator = tmp_path / "s2Dv2ProPlus.pth"
    generator.touch()
    discriminator.touch()
    values = {
        "preprocess_dir": preprocess,
        "output_dir": tmp_path / "output",
        "base_s2g_path": generator,
        "base_s2d_path": discriminator,
        "target_optimizer_steps": 1,
        "checkpoint_every_steps": 1,
    }
    values.update(changes)
    return S2TrainConfig(**values)


def test_s2_config_has_pinned_defaults(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert (config.device, config.precision, config.batch_size, config.num_workers) == ("cuda:0", "fp16", 2, 0)
    assert config.seed == 1234
    assert (config.learning_rate, config.text_low_lr_rate) == (1e-4, 0.4)
    assert config.betas == (0.8, 0.99)
    assert (config.eps, config.lr_decay) == (1e-9, 0.999875)
    assert (config.segment_size, config.c_mel, config.c_kl) == (20480, 45.0, 1.0)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"target_optimizer_steps": 0}, "target_optimizer_steps"),
        ({"checkpoint_every_steps": 0}, "checkpoint_every_steps"),
        ({"batch_size": 0}, "batch_size"),
        ({"num_workers": -1}, "num_workers"),
        ({"precision": "bf16"}, "precision"),
        ({"device": "cpu", "precision": "fp16"}, "fp16"),
    ],
)
def test_s2_config_rejects_invalid_values(tmp_path: Path, changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, **changes).validate()


def test_s2_config_rejects_wrong_checkpoint_names(tmp_path: Path) -> None:
    wrong = tmp_path / "generator.pth"
    wrong.touch()
    with pytest.raises(ValueError, match="s2Gv2ProPlus"):
        _config(tmp_path, base_s2g_path=wrong).validate()


def test_s2_config_rejects_missing_inputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.base_s2d_path.unlink()
    with pytest.raises(ValueError, match="base_s2d_path"):
        config.validate()


def test_s2_config_rejects_cuda_when_unavailable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        _config(tmp_path).validate()
