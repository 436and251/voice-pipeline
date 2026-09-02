"""Production adapter for the GPT-SoVITS Japanese frontend."""

from .japanese import g2p, text_normalize


class JapaneseFrontend:
    def process(self, text: str) -> tuple[str, list[str], None]:
        normalized = text_normalize(text)
        return normalized, g2p(normalized, with_prosody=True), None
