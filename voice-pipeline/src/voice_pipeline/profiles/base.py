from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    sample_rate: int
    semantic_frame_rate: str
    requires_sv: bool
    s1_relative_path: str
    s2g_relative_path: str
    s2d_relative_path: str
    bert_relative_path: str
    hubert_relative_path: str
    speaker_relative_path: str
    s2_family: str
    s2_train_entry: str
    generator_class: str
    discriminator_class: str
    uses_sv_embedding: bool
    uses_gan_training: bool
    uses_cfm_training: bool
    uses_external_vocoder: bool
    text_low_lr_rate: float
