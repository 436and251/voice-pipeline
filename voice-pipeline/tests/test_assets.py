from pathlib import Path

from voice_pipeline.common.assets import verify_profile_assets
from voice_pipeline.profiles.registry import ProfileRegistry


def test_asset_verification_reports_missing_items(tmp_path: Path):
    profile = ProfileRegistry.get("v2ProPlus")
    s1 = tmp_path / profile.s1_relative_path
    s1.parent.mkdir(parents=True)
    s1.write_bytes(b"x")

    checks = verify_profile_assets(profile, tmp_path)
    by_name = {item.name: item for item in checks}

    assert by_name["s1"].exists is True
    assert by_name["s2g"].exists is False
    assert by_name["s2d"].exists is False
    assert by_name["bert"].exists is False
    assert by_name["hubert"].exists is False
    assert by_name["speaker"].exists is False
    assert by_name["g2pw"].path == tmp_path / profile.g2pw_relative_path
    assert by_name["nltk"].path == tmp_path / profile.nltk_data_relative_path
    assert by_name["langdetect"].path == tmp_path / profile.langdetect_relative_path
