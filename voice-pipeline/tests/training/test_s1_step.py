from pathlib import Path

import pytest
import torch

from voice_pipeline.core.gpt_sovits.s1.lr_scheduler import FixedS1LRSchedule
from voice_pipeline.core.gpt_sovits.s1.optim import ScaledAdam
from voice_pipeline.training.s1.config import S1TrainConfig
from voice_pipeline.training.s1.optim import build_optimizer
from voice_pipeline.training.s1.step import backward_s1_minibatch, finish_s1_optimizer_step


class TinyS1(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.called = 0

    def forward_old(self, phoneme_ids, phoneme_ids_len, semantic_ids, semantic_ids_len, bert_feature):
        self.called += 1
        return self.weight.sum() * phoneme_ids.float().sum(), 0.75

    def forward(self, *args):
        raise AssertionError("DPO forward must not be used")


def _config(tmp_path: Path) -> S1TrainConfig:
    return S1TrainConfig(
        preprocess_dir=tmp_path,
        output_dir=tmp_path,
        base_s1_path=tmp_path / "s1v3.ckpt",
        target_optimizer_steps=1,
        checkpoint_every_steps=1,
        device="cpu",
        precision="fp32",
    )


def _batch() -> dict[str, torch.Tensor | list[str]]:
    return {
        "sample_ids": ["fixed"],
        "phoneme_ids": torch.tensor([[1, 2]]),
        "phoneme_ids_len": torch.tensor([2]),
        "semantic_ids": torch.tensor([[3, 4, 5]]),
        "semantic_ids_len": torch.tensor([3]),
        "bert_feature": torch.zeros(1, 1024, 2),
    }


def test_builder_uses_all_named_parameters_and_official_values(tmp_path: Path) -> None:
    model = TinyS1()
    optimizer = build_optimizer(model, _config(tmp_path))

    assert isinstance(optimizer, ScaledAdam)
    assert optimizer.parameters_names == [["weight"]]
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)
    assert optimizer.param_groups[0]["clipping_scale"] == 2.0
    assert optimizer.param_groups[0]["clipping_update_period"] == 1000


def test_four_backwards_sum_losses_before_one_update(tmp_path: Path) -> None:
    model = TinyS1()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = FixedS1LRSchedule(optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    optimizer.zero_grad()
    before = model.weight.detach().clone()

    for _ in range(4):
        result = backward_s1_minibatch(_batch(), model, scaler, _config(tmp_path))
        assert result.loss == pytest.approx(9.0)
        assert result.top3_accuracy == pytest.approx(0.75)

    assert model.called == 4
    assert torch.equal(model.weight, before)
    assert torch.equal(model.weight.grad, torch.tensor([12.0, 12.0]))
    update = finish_s1_optimizer_step(model, optimizer, scheduler, scaler)
    assert not torch.equal(model.weight, before)
    assert update.learning_rate == 0.01
    assert optimizer.param_groups[0]["lr"] == 0.002
    assert model.weight.grad is None
