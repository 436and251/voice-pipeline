from dataclasses import dataclass
from pathlib import Path
from voice_pipeline.profiles.base import ModelProfile


@dataclass(frozen=True, slots=True)
class AssetCheck:
    name: str
    path: Path
    exists: bool


def verify_profile_assets(profile: ModelProfile, project_root: Path) -> list[AssetCheck]:
    assets = {
        "s1": profile.s1_relative_path,
        "s2g": profile.s2g_relative_path,
        "s2d": profile.s2d_relative_path,
        "bert": profile.bert_relative_path,
        "hubert": profile.hubert_relative_path,
        "speaker": profile.speaker_relative_path,
        "g2pw": profile.g2pw_relative_path,
        "nltk": profile.nltk_data_relative_path,
        "langdetect": profile.langdetect_relative_path,
    }
    return [AssetCheck(name, project_root / rel, (project_root / rel).exists()) for name, rel in assets.items()]
