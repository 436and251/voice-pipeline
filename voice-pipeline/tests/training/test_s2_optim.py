from pathlib import Path

import pytest
import torch

from voice_pipeline.training.s2.config import S2TrainConfig
from voice_pipeline.training.s2.optim import build_optimizers, build_schedulers


class TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_embedding = torch.nn.Linear(2, 2)
        self.encoder_text = torch.nn.Linear(2, 2)
        self.mrte = torch.nn.Linear(2, 2)


class TinyGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc_p = TinyEncoder()
        self.base = torch.nn.Linear(2, 2)
        self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)


def _config(tmp_path: Path) -> S2TrainConfig:
    return S2TrainConfig(
        preprocess_dir=tmp_path,
        output_dir=tmp_path,
        base_s2g_path=tmp_path / "s2Gv2ProPlus.pth",
        base_s2d_path=tmp_path / "s2Dv2ProPlus.pth",
        target_optimizer_steps=1,
        checkpoint_every_steps=1,
    )


def test_generator_optimizer_groups_are_disjoint_and_exhaustive(tmp_path: Path) -> None:
    generator = TinyGenerator()
    discriminator = torch.nn.Linear(2, 1)
    optim_g, optim_d = build_optimizers(generator, discriminator, _config(tmp_path))

    assert len(optim_g.param_groups) == 4
    assert [group["lr"] for group in optim_g.param_groups] == [1e-4, 4e-5, 4e-5, 4e-5]
    grouped = [parameter for group in optim_g.param_groups for parameter in group["params"]]
    expected = [parameter for parameter in generator.parameters() if parameter.requires_grad]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in expected}
    assert all(id(generator.frozen) != id(parameter) for parameter in grouped)
    assert optim_g.defaults["betas"] == (0.8, 0.99)
    assert optim_g.defaults["eps"] == 1e-9
    assert optim_d.param_groups[0]["lr"] == 1e-4


def test_optimizer_rejects_shared_low_lr_parameter(tmp_path: Path) -> None:
    generator = TinyGenerator()
    generator.enc_p.encoder_text = generator.enc_p.text_embedding
    with pytest.raises(ValueError, match="duplicate"):
        build_optimizers(generator, torch.nn.Linear(2, 1), _config(tmp_path))


def test_schedulers_use_pinned_epoch_decay(tmp_path: Path) -> None:
    optimizers = build_optimizers(TinyGenerator(), torch.nn.Linear(2, 1), _config(tmp_path))
    schedulers = build_schedulers(*optimizers, _config(tmp_path))
    assert [scheduler.gamma for scheduler in schedulers] == [0.999875, 0.999875]
