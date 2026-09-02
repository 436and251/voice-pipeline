from __future__ import annotations

from dataclasses import astuple
from pathlib import Path

import pytest
import torch

from voice_pipeline.training.s2.config import S2TrainConfig
from voice_pipeline.training.s2.step import train_s2_step


class TinyGenerator(torch.nn.Module):
    def __init__(self, events: list) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.events = events

    def forward(self, ssl, spec, spec_lengths, text, text_lengths, sv_emb):
        self.events.append(("generator", ssl.requires_grad, text.dtype, sv_emb.shape))
        value = self.weight
        y_hat = value.expand(1, 1, 4)
        latent = value.expand(1, 1, 2)
        mask = torch.ones_like(latent)
        return y_hat, value.square(), torch.tensor([0]), mask, mask, (
            latent, latent, latent, latent, latent, latent
        ), latent


class TinyDiscriminator(torch.nn.Module):
    def __init__(self, events: list) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.events = events

    def forward(self, real, generated):
        self.events.append(("discriminator", generated.requires_grad))
        real_score = [real.mean((1, 2), keepdim=True) * self.weight]
        fake_score = [generated.mean((1, 2), keepdim=True) * self.weight]
        return real_score, fake_score, [[real * self.weight]], [[generated * self.weight]]


class RecordingAdamW(torch.optim.AdamW):
    def __init__(self, name: str, parameters, events: list) -> None:
        super().__init__(parameters, lr=0.01)
        self.name = name
        self.events = events

    def zero_grad(self, *args, **kwargs):
        self.events.append(("zero", self.name))
        return super().zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.events.append(("step", self.name))
        return super().step(*args, **kwargs)


class _ScaledLoss:
    def __init__(self, loss: torch.Tensor, events: list) -> None:
        self.loss = loss
        self.events = events

    def backward(self) -> None:
        self.events.append("backward")
        self.loss.backward()


class RecordingScaler:
    def __init__(self, events: list, fail_on_step: int | None = None) -> None:
        self.events = events
        self.steps = 0
        self.fail_on_step = fail_on_step

    def scale(self, loss):
        self.events.append("scale")
        return _ScaledLoss(loss, self.events)

    def unscale_(self, optimizer):
        self.events.append(("unscale", optimizer.name))

    def step(self, optimizer):
        self.steps += 1
        if self.fail_on_step == self.steps:
            raise RuntimeError("optimizer failed")
        optimizer.step()

    def update(self):
        self.events.append("update")


def _config(tmp_path: Path) -> S2TrainConfig:
    return S2TrainConfig(
        preprocess_dir=tmp_path,
        output_dir=tmp_path,
        base_s2g_path=tmp_path / "s2Gv2ProPlus.pth",
        base_s2d_path=tmp_path / "s2Dv2ProPlus.pth",
        target_optimizer_steps=1,
        checkpoint_every_steps=1,
        device="cpu",
        precision="fp32",
        segment_size=4,
        c_mel=2,
        c_kl=3,
    )


def _batch():
    return (
        torch.ones(1, 768, 2),
        torch.tensor([2]),
        torch.ones(1, 1025, 4),
        torch.tensor([4]),
        torch.ones(1, 1, 4),
        torch.tensor([4]),
        torch.ones(1, 2, dtype=torch.long),
        torch.tensor([2]),
        torch.ones(1, 20480),
    )


def _patch_losses(monkeypatch) -> None:
    monkeypatch.setattr("voice_pipeline.training.s2.step.spec_to_mel_torch", lambda spec, *args: spec[:, :1])
    monkeypatch.setattr("voice_pipeline.training.s2.step.mel_spectrogram_torch", lambda generated, *args, **kwargs: generated.unsqueeze(1))
    monkeypatch.setattr("voice_pipeline.training.s2.step.slice_segments", lambda value, ids, size: value)
    monkeypatch.setattr(
        "voice_pipeline.training.s2.step.discriminator_loss",
        lambda real, fake: (sum(item.mean() for item in real + fake), [], []),
    )
    monkeypatch.setattr(
        "voice_pipeline.training.s2.step.generator_loss",
        lambda fake: (sum(x.mean() * 0 for x in fake) + 0.0625, []),
    )
    monkeypatch.setattr(
        "voice_pipeline.training.s2.step.feature_loss",
        lambda real, fake: sum((left[0] - right[0]).abs().mean() for left, right in zip(real, fake)),
    )
    monkeypatch.setattr("voice_pipeline.training.s2.step.kl_loss", lambda *args: args[0].square().mean())


def test_step_runs_detached_d_then_attached_g_and_reports_metrics(tmp_path: Path, monkeypatch) -> None:
    _patch_losses(monkeypatch)
    events: list = []
    generator = TinyGenerator(events)
    discriminator = TinyDiscriminator(events)
    optim_g = RecordingAdamW("g", generator.parameters(), events)
    optim_d = RecordingAdamW("d", discriminator.parameters(), events)
    result = train_s2_step(
        batch=_batch(),
        net_g=generator,
        net_d=discriminator,
        optim_g=optim_g,
        optim_d=optim_d,
        scaler=RecordingScaler(events),
        config=_config(tmp_path),
    )

    assert events[0] == ("generator", False, torch.int64, torch.Size([1, 20480]))
    assert [event for event in events if isinstance(event, tuple) and event[0] == "discriminator"] == [
        ("discriminator", False),
        ("discriminator", True),
    ]
    assert events.index(("step", "d")) < events.index(("zero", "g")) < events.index(("step", "g"))
    assert events[-1] == "update"
    assert result.generator_total == pytest.approx(result.feature + result.mel + result.kl_ssl + result.kl + 0.0625)
    assert all(isinstance(value, float) for value in astuple(result))


def test_step_propagates_generator_optimizer_failure(tmp_path: Path, monkeypatch) -> None:
    _patch_losses(monkeypatch)
    events: list = []
    generator = TinyGenerator(events)
    discriminator = TinyDiscriminator(events)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        train_s2_step(
            batch=_batch(),
            net_g=generator,
            net_d=discriminator,
            optim_g=RecordingAdamW("g", generator.parameters(), events),
            optim_d=RecordingAdamW("d", discriminator.parameters(), events),
            scaler=RecordingScaler(events, fail_on_step=2),
            config=_config(tmp_path),
        )
    assert "update" not in events
