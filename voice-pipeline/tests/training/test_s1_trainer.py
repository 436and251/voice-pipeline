from __future__ import annotations

import json
from pathlib import Path

import torch

from voice_pipeline.common.logging import PipelineLogger
from voice_pipeline.core.gpt_sovits.s1 import FixedS1LRSchedule
from voice_pipeline.training.s1 import S1TrainConfig, S1Trainer, S1TrainingCursor


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


def _batch() -> dict[str, torch.Tensor | list[str]]:
    return {
        "sample_ids": ["fixed"], "phoneme_ids": torch.ones(1, 1, dtype=torch.long),
        "phoneme_ids_len": torch.ones(1, dtype=torch.long),
        "semantic_ids": torch.ones(1, 2, dtype=torch.long),
        "semantic_ids_len": torch.tensor([2]), "bert_feature": torch.zeros(1, 1024, 1),
    }


def _trainer(tmp_path: Path, batch_count: int, target: int) -> S1Trainer:
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
        cleanup=lambda *_: None,
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
