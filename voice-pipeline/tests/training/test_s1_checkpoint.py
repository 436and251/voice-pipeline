from pathlib import Path
import random

import pytest
import torch

from voice_pipeline.core.gpt_sovits.s1 import FixedS1LRSchedule
from voice_pipeline.training.s1.checkpoint import (
    S1TrainingCursor,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from voice_pipeline.training.s1.config import S1TrainConfig
from voice_pipeline.training.s1.optim import build_optimizer


def _objects(tmp_path: Path):
    model = torch.nn.Linear(2, 2)
    config = S1TrainConfig(tmp_path, tmp_path, tmp_path / "s1v3.ckpt", 3, 1, device="cpu", precision="fp32")
    optimizer = build_optimizer(model, config)
    scheduler = FixedS1LRSchedule(optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler, scaler


def test_checkpoint_round_trip_restores_all_training_state(tmp_path: Path) -> None:
    model, optimizer, scheduler, scaler = _objects(tmp_path)
    cursor = S1TrainingCursor(1, 2, 3, 0)
    path = checkpoint_path(tmp_path, 1)
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, cursor=cursor)
    assert path.is_file() and not list(path.parent.glob("*.tmp"))

    fresh_model, fresh_optimizer, fresh_scheduler, fresh_scaler = _objects(tmp_path)
    restored = load_checkpoint(
        path,
        model=fresh_model,
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        scaler=fresh_scaler,
        target_optimizer_steps=3,
    )
    assert restored == cursor
    assert all(torch.equal(fresh_model.state_dict()[name], value) for name, value in expected.items())
    assert fresh_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
    assert fresh_scheduler.state_dict() == scheduler.state_dict()


def test_checkpoint_rejects_partial_accumulation(tmp_path: Path) -> None:
    model, optimizer, scheduler, scaler = _objects(tmp_path)
    with pytest.raises(ValueError, match="accumulation_position"):
        save_checkpoint(
            checkpoint_path(tmp_path, 1),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cursor=S1TrainingCursor(1, 0, 0, 1),
        )


def test_late_corruption_does_not_mutate_live_model(tmp_path: Path) -> None:
    model, optimizer, scheduler, scaler = _objects(tmp_path)
    path = checkpoint_path(tmp_path, 1)
    save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        cursor=S1TrainingCursor(1, 0, 0, 0),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["scheduler"] = {"current_step": -1}
    torch.save(payload, path)
    fresh_model, fresh_optimizer, fresh_scheduler, fresh_scaler = _objects(tmp_path)
    before = {name: value.clone() for name, value in fresh_model.state_dict().items()}

    with pytest.raises(ValueError, match="scheduler"):
        load_checkpoint(
            path, model=fresh_model, optimizer=fresh_optimizer, scheduler=fresh_scheduler,
            scaler=fresh_scaler, target_optimizer_steps=3,
        )
    assert all(torch.equal(fresh_model.state_dict()[name], value) for name, value in before.items())


def test_scaled_adam_tensor_shape_corruption_is_rejected_before_mutation(tmp_path: Path) -> None:
    model, optimizer, scheduler, scaler = _objects(tmp_path)
    path = checkpoint_path(tmp_path, 1)
    save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        cursor=S1TrainingCursor(1, 0, 0, 0),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = next(iter(payload["optimizer"]["state"].values()))
    state["delta"] = state["delta"].reshape(-1)
    torch.save(payload, path)
    fresh_model, fresh_optimizer, fresh_scheduler, fresh_scaler = _objects(tmp_path)
    before = {name: value.clone() for name, value in fresh_model.state_dict().items()}

    with pytest.raises(ValueError, match="optimizer tensor state"):
        load_checkpoint(
            path, model=fresh_model, optimizer=fresh_optimizer, scheduler=fresh_scheduler,
            scaler=fresh_scaler, target_optimizer_steps=3,
        )
    assert all(torch.equal(fresh_model.state_dict()[name], value) for name, value in before.items())


def test_duplicate_optimizer_parameter_identifier_is_rejected(tmp_path: Path) -> None:
    model, optimizer, scheduler, scaler = _objects(tmp_path)
    path = checkpoint_path(tmp_path, 1)
    save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        cursor=S1TrainingCursor(1, 0, 0, 0),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    parameters = payload["optimizer"]["param_groups"][0]["params"]
    parameters[1] = parameters[0]
    torch.save(payload, path)

    with pytest.raises(ValueError, match="duplicate S1 optimizer parameter identifier"):
        load_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, target_optimizer_steps=3,
        )


def test_incomplete_scaled_adam_state_is_rejected(tmp_path: Path) -> None:
    model, optimizer, scheduler, scaler = _objects(tmp_path)
    path = checkpoint_path(tmp_path, 1)
    save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        cursor=S1TrainingCursor(1, 0, 0, 0),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del next(iter(payload["optimizer"]["state"].values()))["delta"]
    torch.save(payload, path)

    with pytest.raises(ValueError, match="optimizer state keys"):
        load_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, target_optimizer_steps=3,
        )


def test_batched_scalar_scaled_adam_state_requires_scale_fields(tmp_path: Path) -> None:
    class TwinScalar(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = torch.nn.Parameter(torch.tensor(1.0))
            self.right = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self):
            return self.left + self.right

    model = TwinScalar()
    config = S1TrainConfig(tmp_path, tmp_path, tmp_path / "s1v3.ckpt", 3, 1, device="cpu", precision="fp32")
    optimizer = build_optimizer(model, config)
    scheduler = FixedS1LRSchedule(optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model().backward()
    optimizer.step()
    scheduler.step()
    path = checkpoint_path(tmp_path, 1)
    save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        cursor=S1TrainingCursor(1, 0, 0, 0),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del next(iter(payload["optimizer"]["state"].values()))["param_rms"]
    torch.save(payload, path)

    with pytest.raises(ValueError, match="optimizer state keys"):
        load_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, target_optimizer_steps=3,
        )


@pytest.mark.gpu
def test_cuda_rng_corruption_is_rejected_before_mutation(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to validate CUDA RNG state")
    model, optimizer, scheduler, scaler = _objects(tmp_path)
    path = checkpoint_path(tmp_path, 1)
    save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        cursor=S1TrainingCursor(1, 0, 0, 0),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["cuda_rng"][0] = torch.tensor([1], dtype=torch.uint8)
    torch.save(payload, path)
    fresh_model, fresh_optimizer, fresh_scheduler, fresh_scaler = _objects(tmp_path)
    before = {name: value.clone() for name, value in fresh_model.state_dict().items()}

    with pytest.raises(ValueError, match="CUDA RNG"):
        load_checkpoint(
            path, model=fresh_model, optimizer=fresh_optimizer, scheduler=fresh_scheduler,
            scaler=fresh_scaler, target_optimizer_steps=3,
        )
    assert all(torch.equal(fresh_model.state_dict()[name], value) for name, value in before.items())


def test_checkpoint_restores_rng(tmp_path: Path) -> None:
    model, optimizer, scheduler, scaler = _objects(tmp_path)
    path = checkpoint_path(tmp_path, 1)
    random.seed(7)
    torch.manual_seed(7)
    save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, cursor=S1TrainingCursor(1, 0, 0, 0))
    expected = (random.random(), torch.rand(1))
    load_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, target_optimizer_steps=2)
    assert random.random() == expected[0]
    assert torch.equal(torch.rand(1), expected[1])
