from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from voice_pipeline.common.logging import PipelineLogger
from voice_pipeline.core.gpt_sovits.s1 import FixedS1LRSchedule
from voice_pipeline.training.s1 import S1TrainConfig, S1Trainer, S1TrainingCursor, load_checkpoint
from voice_pipeline.training.s1.optim import build_optimizer


class TinyS1(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward_old(self, phoneme_ids, phoneme_ids_len, semantic_ids, semantic_ids_len, bert_feature):
        return self.weight.sum() * phoneme_ids.float().sum(), 0.5


class EpochSampler:
    def __init__(self) -> None:
        self.epochs = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return _batch()


def _batch() -> dict[str, torch.Tensor | list[str]]:
    return {
        "sample_ids": ["fixed"], "phoneme_ids": torch.ones(1, 1, dtype=torch.long),
        "phoneme_ids_len": torch.ones(1, dtype=torch.long),
        "semantic_ids": torch.ones(1, 2, dtype=torch.long),
        "semantic_ids_len": torch.tensor([2]), "bert_feature": torch.zeros(1, 1024, 1),
    }


def _trainer(tmp_path: Path, batch_count: int, target: int, *, cleanup=lambda *_: None) -> S1Trainer:
    model = TinyS1()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return S1Trainer(
        config=S1TrainConfig(tmp_path, tmp_path / "out", tmp_path / "s1v3.ckpt", target, 1, device="cpu", precision="fp32"),
        model=model,
        optimizer=optimizer,
        scheduler=FixedS1LRSchedule(optimizer),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        loader=[_batch() for _ in range(batch_count)],
        sampler=EpochSampler(),
        logger=PipelineLogger(tmp_path / "events.jsonl", echo=False),
        cursor=S1TrainingCursor(),
        cleanup=cleanup,
    )


def _scaled_trainer(tmp_path: Path, batch_count: int, target: int) -> S1Trainer:
    model = TinyS1()
    config = S1TrainConfig(
        tmp_path, tmp_path / "out", tmp_path / "s1v3.ckpt", target, 1,
        device="cpu", precision="fp32",
    )
    optimizer = build_optimizer(model, config)
    return S1Trainer(
        config=config, model=model, optimizer=optimizer,
        scheduler=FixedS1LRSchedule(optimizer),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        loader=[_batch() for _ in range(batch_count)], sampler=EpochSampler(),
        logger=PipelineLogger(tmp_path / "events.jsonl", echo=False),
        cursor=S1TrainingCursor(), cleanup=lambda *_: None,
    )


def test_eight_mini_batches_produce_exactly_two_optimizer_steps(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, 8, 2)
    cursor = trainer.train()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]

    assert cursor.optimizer_step == 2
    assert len([event for event in events if event["event"] == "mini_batch"]) == 8
    optimizer_events = [event for event in events if event["event"] == "optimizer"]
    assert len(optimizer_events) == 2
    assert [event["metrics"]["learning_rate"] for event in optimizer_events] == [0.01, 0.002]


def test_accumulation_continues_across_epoch_boundary(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, 3, 1)
    cursor = trainer.train()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]

    assert cursor == S1TrainingCursor(1, 1, 1, 0)
    assert len([event for event in events if event["event"] == "mini_batch"]) == 4
    assert trainer.sampler.epochs == [0, 1]


def test_final_checkpoint_restores_at_next_batch(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, 5, 1)
    cursor = trainer.train()
    checkpoint = tmp_path / "out" / "training" / "s1" / "checkpoints" / "step-00000001.pt"
    assert checkpoint.is_file()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["optimizer_step"] == 1
    assert payload["next_batch_index"] == cursor.next_batch_index == 4
    assert payload["accumulation_position"] == 0


def test_resume_reproduces_uninterrupted_weights(tmp_path: Path) -> None:
    uninterrupted = _scaled_trainer(tmp_path / "full", 5, 2)
    uninterrupted.train()

    first = _scaled_trainer(tmp_path / "resumed", 5, 1)
    first.train()
    resumed = _scaled_trainer(tmp_path / "resumed", 5, 2)
    resumed.cursor = load_checkpoint(
        first.config.output_dir / "training/s1/checkpoints/step-00000001.pt",
        model=resumed.model, optimizer=resumed.optimizer, scheduler=resumed.scheduler,
        scaler=resumed.scaler, target_optimizer_steps=2,
    )
    resumed.train()

    assert resumed.cursor == uninterrupted.cursor
    assert torch.equal(resumed.model.weight, uninterrupted.model.weight)


def test_cleanup_runs_only_after_successful_final_checkpoint(tmp_path: Path, monkeypatch) -> None:
    cleanup_calls = []
    successful = _trainer(tmp_path / "success", 4, 1, cleanup=lambda *args: cleanup_calls.append(args))
    successful.train()
    assert cleanup_calls == [(successful.config.preprocess_dir, True)]

    cleanup_calls.clear()
    failing = _trainer(tmp_path / "failure", 4, 1, cleanup=lambda *args: cleanup_calls.append(args))
    monkeypatch.setattr(
        "voice_pipeline.training.s1.trainer.save_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        failing.train()
    assert cleanup_calls == []


def test_minibatch_exception_writes_no_checkpoint_and_performs_no_cleanup(tmp_path: Path) -> None:
    cleanup_calls = []
    trainer = _trainer(tmp_path, 1, 1, cleanup=lambda *args: cleanup_calls.append(args))
    trainer.loader = [{}]

    with pytest.raises(KeyError):
        trainer.train()
    assert not (tmp_path / "out/training/s1/checkpoints").exists()
    assert cleanup_calls == []


def test_dataloader_iteration_uses_an_independent_rng(tmp_path: Path, monkeypatch) -> None:
    preprocess = tmp_path / "preprocess"
    preprocess.mkdir()
    base_s1 = tmp_path / "s1v3.ckpt"
    base_s1.touch()
    config = S1TrainConfig(
        preprocess, tmp_path / "out", base_s1, 1, 1,
        device="cpu", precision="fp32", batch_size=1,
    )
    monkeypatch.setattr("voice_pipeline.training.s1.trainer.load_s1_checkpoint", lambda *_: TinyS1())
    monkeypatch.setattr("voice_pipeline.training.s1.trainer.S1Dataset", lambda *args, **kwargs: TinyDataset())
    monkeypatch.setattr("voice_pipeline.training.s1.trainer.S1Collate", lambda: lambda items: items[0])
    trainer = S1Trainer.from_pretrained(config)
    before = torch.random.get_rng_state()

    next(iter(trainer.loader))

    assert torch.equal(torch.random.get_rng_state(), before)


def test_keyboard_interrupt_writes_no_checkpoint_and_performs_no_cleanup(tmp_path: Path) -> None:
    class InterruptingLoader:
        def __len__(self):
            return 1

        def __iter__(self):
            raise KeyboardInterrupt

    cleanup_calls = []
    trainer = _trainer(tmp_path, 1, 1, cleanup=lambda *args: cleanup_calls.append(args))
    trainer.loader = InterruptingLoader()

    with pytest.raises(KeyboardInterrupt):
        trainer.train()
    assert not (tmp_path / "out/training/s1/checkpoints").exists()
    assert cleanup_calls == []
