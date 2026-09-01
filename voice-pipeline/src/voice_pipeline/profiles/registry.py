from .base import ModelProfile
from .v2proplus import V2PROPLUS

PROFILE_REGISTRY = {V2PROPLUS.name: V2PROPLUS}


class ProfileRegistry:
    _profiles = PROFILE_REGISTRY

    @classmethod
    def get(cls, name: str) -> ModelProfile:
        try:
            return cls._profiles[name]
        except KeyError as exc:
            raise KeyError(f"unsupported GPT-SoVITS profile: {name}") from exc


def get_profile(name: str) -> ModelProfile:
    return ProfileRegistry.get(name)
