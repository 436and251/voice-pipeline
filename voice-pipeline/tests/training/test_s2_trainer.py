from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from voice_pipeline.common.logging import PipelineLogger
from voice_pipeline.training.s2.checkpoint import TrainingCursor, checkpoint_path
from voice_pipeline.training.s2.config import S2TrainConfig
from voice_pipeline.training.s2.data import DeterministicEpochSampler
from voice_pipeline.training.s2.step import S2StepResult
from voice_pipeline.training.s2.trainer import S2Trainer


class TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_embedding = torch.nn.Linear(1, 1)
        self.encoder_text = torch.nn.Linear(1, 1)
        self.mrte = torch.nn.Linear(1, 1)


class TinyGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc_p = TinyEncoder()
        self.base = torch.nn.Linear(1, 1)


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return tuple(torch.tensor(index) for _ in range(9))


class FirstOnlyCollate:
    def __call__(self, batch):
        return batch[0]


RESULT = S2StepResult(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)


def _config(tmp_path: Path, *, target: int = 1, interval: int = 1) -> S2TrainConfig:
    preprocess = tmp_path / "preprocess"
    preprocess.mkdir(parents=True, exist_ok=True)
    generator = tmp_path / "s2Gv2ProPlus.pth"
    discriminator = tmp_path / "s2Dv2ProPlus.pth"
    generator.touch(exist_ok=True)
    discriminator.touch(exist_ok=True)
    return S2TrainConfig(
        preprocess_dir=preprocess,
        output_dir=tmp_path / "output",
        base_s2g_path=generator,
        base_s2d_path=discriminator,
        target_optimizer_steps=target,
        checkpoint_every_steps=interval,
        device="cpu",
        precision="fp32",
        batch_size=1,
    )


def _patch_construction(monkeypatch, calls: list) -> None:
    monkeypatch.setattr("voice_pipeline.training.s2.trainer.S2Dataset", lambda path: TinyDataset())
    monkeypatch.setattr("voice_pipeline.training.s2.trainer.S2Collate", FirstOnlyCollate)

    def load_g(path, device):
        calls.append(("load_g", path, str(device)))
        return TinyGenerator().to(device)

    def load_d(path, device):
        calls.append(("load_d", path, str(device)))
        return torch.nn.Linear(1, 1).to(device)

    monkeypatch.setattr("voice_pipeline.training.s2.trainer.load_s2_generator", load_g)
    monkeypatch.setattr("voice_pipeline.training.s2.trainer.load_s2_discriminator", load_d)


def test_from_pretrained_validates_before_loading(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    _patch_construction(monkeypatch, calls)
    trainer = S2Trainer.from_pretrained(_config(tmp_path))
    assert [call[0] for call in calls] == ["load_g", "load_d"]
    assert trainer.cursor == TrainingCursor()

    invalid = _config(tmp_path / "invalid")
    invalid.base_s2g_path.unlink()
    with pytest.raises(ValueError):
        S2Trainer.from_pretrained(invalid)
    assert len(calls) == 2


def test_one_step_logs_checkpoints_and_cleans_only_after_success(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    _patch_construction(monkeypatch, calls)
    monkeypatch.setattr("voice_pipeline.training.s2.trainer.train_s2_step", lambda **kwargs: RESULT)
    monkeypatch.setattr(
        "voice_pipeline.training.s2.trainer.cleanup_after_training",
        lambda path, succeeded: calls.append(("cleanup", path, succeeded)),
    )
    log_path = tmp_path / "events.jsonl"
    trainer = S2Trainer.from_pretrained(_config(tmp_path), logger=PipelineLogger(log_path, echo=False))
    cursor = trainer.train()

    assert cursor == TrainingCursor(1, 0, 1)
    assert checkpoint_path(trainer.config.output_dir, 1).is_file()
    assert not list(checkpoint_path(trainer.config.output_dir, 1).parent.glob("*.tmp"))
    assert calls[-1] == ("cleanup", trainer.config.preprocess_dir, True)
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["batch", "checkpoint"]
    assert events[0]["optimizer_step"] == 1
    assert events[0]["metrics"]["generator_total"] == 2.0


def test_resume_skips_saved_batch_and_steps_schedulers_at_epoch_end(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    batches: list[int] = []
    _patch_construction(monkeypatch, calls)
    monkeypatch.setattr("voice_pipeline.training.s2.trainer.cleanup_after_training", lambda *args: None)
    monkeypatch.setattr("voice_pipeline.training.s2.trainer.train_s2_step", lambda **kwargs: RESULT)
    first = S2Trainer.from_pretrained(_config(tmp_path))
    first.train()
    resume = checkpoint_path(first.config.output_dir, 1)

    def record_step(**kwargs):
        batches.append(int(kwargs["batch"][0]))
        kwargs["optim_d"].step()
        kwargs["optim_g"].step()
        return RESULT

    monkeypatch.setattr("voice_pipeline.training.s2.trainer.train_s2_step", record_step)
    second = S2Trainer.from_pretrained(_config(tmp_path, target=2), resume_from=resume)
    cursor = second.train()
    expected_order = list(DeterministicEpochSampler(TinyDataset(), seed=1234))
    assert batches == [expected_order[1]]
    assert cursor == TrainingCursor(2, 1, 0)
    assert second.scheduler_g.last_epoch == 1
    assert second.scheduler_d.last_epoch == 1


def test_step_failure_writes_nothing_and_does_not_clean(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    _patch_construction(monkeypatch, calls)
    monkeypatch.setattr(
        "voice_pipeline.training.s2.trainer.train_s2_step",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("G failed")),
    )
    monkeypatch.setattr(
        "voice_pipeline.training.s2.trainer.cleanup_after_training",
        lambda *args: calls.append("cleanup"),
    )
    trainer = S2Trainer.from_pretrained(_config(tmp_path))
    with pytest.raises(RuntimeError, match="G failed"):
        trainer.train()
    assert trainer.cursor == TrainingCursor()
    assert not checkpoint_path(trainer.config.output_dir, 1).exists()
    assert "cleanup" not in calls


def test_checkpoint_failure_does_not_clean(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    _patch_construction(monkeypatch, calls)
    monkeypatch.setattr("voice_pipeline.training.s2.trainer.train_s2_step", lambda **kwargs: RESULT)
    monkeypatch.setattr(
        "voice_pipeline.training.s2.trainer.save_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        "voice_pipeline.training.s2.trainer.cleanup_after_training",
        lambda *args: calls.append("cleanup"),
    )
    with pytest.raises(OSError, match="disk full"):
        S2Trainer.from_pretrained(_config(tmp_path)).train()
    assert "cleanup" not in calls
