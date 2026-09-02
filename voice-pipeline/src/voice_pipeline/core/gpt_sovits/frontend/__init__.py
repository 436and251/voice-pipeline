"""Compatibility-preserving GPT-SoVITS text frontends."""

from .contract import FrontendResult

__all__ = ["FrontendResult", "MultilingualFrontend"]


def __getattr__(name: str):
    if name == "MultilingualFrontend":
        from .multilingual import MultilingualFrontend

        return MultilingualFrontend
    raise AttributeError(name)
