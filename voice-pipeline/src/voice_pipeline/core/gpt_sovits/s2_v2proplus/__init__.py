from __future__ import annotations

from .model import MultiPeriodDiscriminator, SynthesizerTrn


def build_s2_generator(config: dict[str, object]) -> SynthesizerTrn:
    data = config["data"]
    train = config["train"]
    model_config = dict(config["model"])
    model_config["version"] = "v2ProPlus"
    return SynthesizerTrn(
        data["filter_length"] // 2 + 1,
        train["segment_size"] // data["hop_length"],
        n_speakers=data["n_speakers"],
        **model_config,
    )


def build_s2_discriminator(*, use_spectral_norm: bool = False) -> MultiPeriodDiscriminator:
    return MultiPeriodDiscriminator(use_spectral_norm=use_spectral_norm, version="v2ProPlus")


__all__ = [
    "MultiPeriodDiscriminator",
    "SynthesizerTrn",
    "build_s2_discriminator",
    "build_s2_generator",
]
