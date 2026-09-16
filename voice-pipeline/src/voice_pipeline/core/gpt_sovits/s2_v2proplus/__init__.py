from __future__ import annotations

def build_s2_generator(config: dict[str, object]):
    from .model import SynthesizerTrn

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


def build_s2_discriminator(*, use_spectral_norm: bool = False):
    from .model import MultiPeriodDiscriminator

    return MultiPeriodDiscriminator(use_spectral_norm=use_spectral_norm, version="v2ProPlus")


def __getattr__(name: str):
    if name not in {"MultiPeriodDiscriminator", "SynthesizerTrn"}:
        raise AttributeError(name)
    from . import model

    value = getattr(model, name)
    globals()[name] = value
    return value


__all__ = [
    "MultiPeriodDiscriminator",
    "SynthesizerTrn",
    "build_s2_discriminator",
    "build_s2_generator",
]
