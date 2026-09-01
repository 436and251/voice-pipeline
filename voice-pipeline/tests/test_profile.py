import pytest

from voice_pipeline.profiles.registry import ProfileRegistry


def test_v2proplus_profile_paths():
    profile = ProfileRegistry.get("v2ProPlus")
    assert profile.name == "v2ProPlus"
    assert profile.sample_rate == 32000
    assert profile.semantic_frame_rate == "25hz"
    assert profile.requires_sv is True
    assert profile.s1_relative_path == "models/pretrained/v2proplus/s1/s1v3.ckpt"
    assert profile.s2g_relative_path.endswith("s2Gv2ProPlus.pth")
    assert profile.s2d_relative_path.endswith("s2Dv2ProPlus.pth")


def test_unknown_profile_rejected():
    with pytest.raises(KeyError):
        ProfileRegistry.get("v4")
