from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from voice_pipeline.core.gpt_sovits.s2_v2proplus.commons import clip_grad_value_, slice_segments
from voice_pipeline.core.gpt_sovits.s2_v2proplus.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from voice_pipeline.core.gpt_sovits.s2_v2proplus.mel_processing import (
    mel_spectrogram_torch,
    spec_to_mel_torch,
)

from .config import S2TrainConfig


@dataclass(frozen=True, slots=True)
class S2StepResult:
    discriminator_total: float
    generator_total: float
    feature: float
    mel: float
    kl_ssl: float
    kl: float
    grad_norm_d: float
    grad_norm_g: float


def train_s2_step(
    *,
    batch,
    net_g,
    net_d,
    optim_g,
    optim_d,
    scaler,
    config: S2TrainConfig,
) -> S2StepResult:
    device = torch.device(config.device)
    ssl, ssl_lengths, spec, spec_lengths, wav, wav_lengths, text, text_lengths, sv_emb = (
        tensor.to(device, non_blocking=device.type == "cuda") for tensor in batch
    )
    del ssl_lengths, wav_lengths
    ssl.requires_grad_(False)
    fp16 = config.precision == "fp16" and device.type == "cuda"

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
        y_hat, kl_ssl_value, ids_slice, _, z_mask, latent, _ = net_g(
            ssl, spec, spec_lengths, text, text_lengths, sv_emb
        )
        z, z_p, m_p, logs_p, m_q, logs_q = latent
        del z, m_q
        mel = spec_to_mel_torch(spec, 2048, 128, 32000, 0.0, None)
        y_mel = slice_segments(mel, ids_slice, config.segment_size // 640)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1), 2048, 128, 32000, 640, 2048, 0.0, None
        )
        y_real = slice_segments(wav, ids_slice * 640, config.segment_size)
        real_score, fake_score, _, _ = net_d(y_real, y_hat.detach())
        with torch.autocast(device_type=device.type, enabled=False):
            loss_d = discriminator_loss(real_score, fake_score)[0]

    optim_d.zero_grad()
    scaler.scale(loss_d).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
        _, fake_score, fmap_real, fmap_fake = net_d(y_real, y_hat)
        with torch.autocast(device_type=device.type, enabled=False):
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * config.c_mel
            loss_kl_value = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.c_kl
            loss_feature = feature_loss(fmap_real, fmap_fake)
            loss_generator = generator_loss(fake_score)[0]
            loss_g = loss_generator + loss_feature + loss_mel + kl_ssl_value + loss_kl_value

    optim_g.zero_grad()
    scaler.scale(loss_g).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()
    return S2StepResult(
        discriminator_total=float(loss_d.detach()),
        generator_total=float(loss_g.detach()),
        feature=float(loss_feature.detach()),
        mel=float(loss_mel.detach()),
        kl_ssl=float(kl_ssl_value.detach()),
        kl=float(loss_kl_value.detach()),
        grad_norm_d=float(grad_norm_d),
        grad_norm_g=float(grad_norm_g),
    )
