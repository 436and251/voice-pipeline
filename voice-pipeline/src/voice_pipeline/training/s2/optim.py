from __future__ import annotations

import torch

from .config import S2TrainConfig


def build_optimizers(net_g, net_d, config: S2TrainConfig) -> tuple[torch.optim.AdamW, torch.optim.AdamW]:
    try:
        low_modules = (net_g.enc_p.text_embedding, net_g.enc_p.encoder_text, net_g.enc_p.mrte)
    except AttributeError as error:
        raise ValueError("generator is missing a v2ProPlus low-LR module") from error
    low_groups = [[parameter for parameter in module.parameters() if parameter.requires_grad] for module in low_modules]
    low_parameters = [parameter for group in low_groups for parameter in group]
    if len(low_parameters) != len({id(parameter) for parameter in low_parameters}):
        raise ValueError("duplicate generator parameter across low-LR groups")

    trainable = [parameter for parameter in net_g.parameters() if parameter.requires_grad]
    low_ids = {id(parameter) for parameter in low_parameters}
    base = [parameter for parameter in trainable if id(parameter) not in low_ids]
    grouped = base + low_parameters
    if len(grouped) != len({id(parameter) for parameter in grouped}) or {id(p) for p in grouped} != {
        id(p) for p in trainable
    }:
        raise ValueError("generator optimizer groups are not disjoint and exhaustive")

    low_lr = config.learning_rate * config.text_low_lr_rate
    optim_g = torch.optim.AdamW(
        [
            {"params": base, "lr": config.learning_rate},
            {"params": low_groups[0], "lr": low_lr},
            {"params": low_groups[1], "lr": low_lr},
            {"params": low_groups[2], "lr": low_lr},
        ],
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )
    optim_d = torch.optim.AdamW(
        (parameter for parameter in net_d.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )
    return optim_g, optim_d


def build_schedulers(
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    config: S2TrainConfig,
) -> tuple[torch.optim.lr_scheduler.ExponentialLR, torch.optim.lr_scheduler.ExponentialLR]:
    return (
        torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=config.lr_decay),
        torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=config.lr_decay),
    )
