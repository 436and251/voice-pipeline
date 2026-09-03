from __future__ import annotations

import pytest
import torch

from voice_pipeline.core.gpt_sovits.s1.lr_scheduler import FixedS1LRSchedule
from voice_pipeline.core.gpt_sovits.s1.optim import ScaledAdam


def test_scaled_adam_updates_named_parameters_with_finite_values() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    second = torch.nn.Parameter(torch.tensor([0.5, -0.5]))
    optimizer = ScaledAdam(
        [first, second],
        lr=0.01,
        betas=(0.9, 0.95),
        clipping_scale=2.0,
        parameters_names=[["first", "second"]],
        show_dominant_parameters=False,
        clipping_update_period=1000,
    )
    before = (first.detach().clone(), second.detach().clone())
    (first.square().sum() + second.square().sum()).backward()
    optimizer.step()

    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert not torch.equal(first, before[0])
    assert not torch.equal(second, before[1])


def test_scaled_adam_matches_upstream_high_rms_updates() -> None:
    first = torch.nn.Parameter(torch.tensor([4.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([4.0, 4.0]))
    optimizer = ScaledAdam(
        [first, second],
        lr=0.01,
        betas=(0.9, 0.95),
        clipping_scale=2.0,
        parameters_names=[["first", "second"]],
        show_dominant_parameters=False,
        clipping_update_period=1000,
    )
    for _ in range(4):
        first.grad = torch.tensor([1.0, 2.0])
        second.grad = torch.tensor([3.0, 4.0])
        optimizer.step()

    assert torch.allclose(first, torch.tensor([3.96223545, 3.96223545]), atol=1e-7)
    assert torch.allclose(second, torch.tensor([3.96223545, 3.96223545]), atol=1e-7)


def test_scaled_adam_matches_upstream_batched_scalar_updates() -> None:
    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(0.5))
    optimizer = ScaledAdam(
        [first, second],
        lr=0.01,
        betas=(0.9, 0.95),
        clipping_scale=2.0,
        parameters_names=[["first", "second"]],
        show_dominant_parameters=False,
        clipping_update_period=1000,
    )
    for _ in range(4):
        first.grad = torch.tensor(1.0)
        second.grad = torch.tensor(2.0)
        optimizer.step()

    assert first.item() == pytest.approx(0.99909502, abs=1e-7)
    assert second.item() == pytest.approx(0.49909514, abs=1e-7)


def test_fixed_s1_scheduler_preserves_actual_upstream_learning_rates() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    scheduler = FixedS1LRSchedule(optimizer)

    assert optimizer.param_groups[0]["lr"] == 0.01
    assert scheduler.step() == 0.002
    assert optimizer.param_groups[0]["lr"] == 0.002
    state = scheduler.state_dict()
    assert scheduler.step() == 0.002

    restored_optimizer = torch.optim.SGD([parameter], lr=0.01)
    restored = FixedS1LRSchedule(restored_optimizer)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored_optimizer.param_groups[0]["lr"] == 0.002


def test_fixed_s1_scheduler_rejects_invalid_state_without_mutation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    scheduler = FixedS1LRSchedule(optimizer)

    for invalid in ({}, {"current_step": True}, {"current_step": -1}, {"current_step": 1, "extra": 2}):
        try:
            scheduler.load_state_dict(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid scheduler state: {invalid}")
        assert scheduler.state_dict() == {"current_step": 0}
        assert optimizer.param_groups[0]["lr"] == 0.01
