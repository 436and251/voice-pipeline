from voice_pipeline.profiles.registry import get_profile


def test_v2proplus_route_is_v2_pro_gan_family():
    profile = get_profile("v2ProPlus")
    assert profile.s2_family == "v2_pro"
    assert profile.s2_train_entry == "s2_train.py"
    assert profile.generator_class == "SynthesizerTrn"
    assert profile.discriminator_class == "MultiPeriodDiscriminator"
    assert profile.uses_sv_embedding is True
    assert profile.uses_gan_training is True
    assert profile.uses_cfm_training is False
    assert profile.uses_external_vocoder is False
    assert profile.text_low_lr_rate == 0.4
