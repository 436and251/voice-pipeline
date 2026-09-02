from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from voice_pipeline.training.s2.checkpoint import (
    TrainingCursor,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)


def _objects():
    net_g = torch.nn.Linear(2, 2)
    net_d = torch.nn.Linear(2, 1)
    optim_g = torch.optim.AdamW(net_g.parameters(), lr=0.01)
    optim_d = torch.optim.AdamW(net_d.parameters(), lr=0.02)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=0.9)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.8)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for net, optimizer, scheduler in (
        (net_g, optim_g, scheduler_g),
        (net_d, optim_d, scheduler_d),
    ):
        optimizer.zero_grad()
        net(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
    return net_g, net_d, optim_g, optim_d, scheduler_g, scheduler_d, scaler


def _save(path: Path, objects, cursor=TrainingCursor(1, 2, 3)) -> None:
    net_g, net_d, optim_g, optim_d, scheduler_g, scheduler_d, scaler = objects
    save_checkpoint(
        path,
        net_g=net_g,
        net_d=net_d,
        optim_g=optim_g,
        optim_d=optim_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        scaler=scaler,
        cursor=cursor,
    )


def _load(path: Path, objects, target=5) -> TrainingCursor:
    net_g, net_d, optim_g, optim_d, scheduler_g, scheduler_d, scaler = objects
    return load_checkpoint(
        path,
        net_g=net_g,
        net_d=net_d,
        optim_g=optim_g,
        optim_d=optim_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        scaler=scaler,
        target_optimizer_steps=target,
    )


def test_checkpoint_path_uses_completed_step() -> None:
    assert checkpoint_path(Path("run"), 1) == Path("run/training/s2/checkpoints/step-00000001.pt")


def test_checkpoint_round_trip_restores_all_state_and_rng(tmp_path: Path) -> None:
    objects = _objects()
    random.seed(7)
    torch.manual_seed(9)
    path = checkpoint_path(tmp_path, 1)
    expected_g = {name: value.clone() for name, value in objects[0].state_dict().items()}
    expected_g_lr = objects[2].param_groups[0]["lr"]
    _save(path, objects)
    expected_python = random.random()
    expected_torch = torch.rand(3)

    with torch.no_grad():
        for parameter in objects[0].parameters():
            parameter.zero_()
    objects[2].param_groups[0]["lr"] = 99
    random.seed(100)
    torch.manual_seed(100)

    assert _load(path, objects) == TrainingCursor(1, 2, 3)
    assert all(torch.equal(value, objects[0].state_dict()[name]) for name, value in expected_g.items())
    assert objects[2].param_groups[0]["lr"] == expected_g_lr
    assert objects[4].last_epoch == 1 and objects[5].last_epoch == 1
    assert random.random() == expected_python
    assert torch.equal(torch.rand(3), expected_torch)
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("net_g"), "net_g"),
        (lambda payload: payload.update(format_version=2), "format_version"),
        (lambda payload: payload.update(profile="v2"), "profile"),
        (lambda payload: payload.update(epoch=-1), "cursor"),
    ],
)
def test_checkpoint_rejects_invalid_envelope(tmp_path: Path, mutation, message: str) -> None:
    objects = _objects()
    path = checkpoint_path(tmp_path, 1)
    _save(path, objects)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mutation(payload)
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        _load(path, objects)


def test_checkpoint_rejects_filename_step_mismatch(tmp_path: Path) -> None:
    objects = _objects()
    path = checkpoint_path(tmp_path, 2)
    _save(path, objects)
    with pytest.raises(ValueError, match="filename"):
        _load(path, objects)


def test_checkpoint_rejects_step_beyond_target(tmp_path: Path) -> None:
    objects = _objects()
    path = checkpoint_path(tmp_path, 6)
    _save(path, objects, TrainingCursor(6, 0, 0))
    with pytest.raises(ValueError, match="target"):
        _load(path, objects, target=5)


def test_checkpoint_loads_models_strictly(tmp_path: Path) -> None:
    objects = _objects()
    path = checkpoint_path(tmp_path, 1)
    _save(path, objects)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["net_g"]["weight"] = torch.zeros(3, 3)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="net_g"):
        _load(path, objects)


def test_late_checkpoint_corruption_does_not_partially_mutate_models(tmp_path: Path) -> None:
    objects = _objects()
    path = checkpoint_path(tmp_path, 1)
    _save(path, objects)
    expected_g = {name: value.clone() for name, value in objects[0].state_dict().items()}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["net_g"] = {name: value + 1 for name, value in payload["net_g"].items()}
    payload["net_d"]["weight"] = torch.zeros(3, 3)
    torch.save(payload, path)

    with pytest.raises((ValueError, RuntimeError)):
        _load(path, objects)
    assert all(torch.equal(value, objects[0].state_dict()[name]) for name, value in expected_g.items())
