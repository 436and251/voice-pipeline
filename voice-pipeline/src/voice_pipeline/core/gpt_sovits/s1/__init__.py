from .models.t2s_model import Text2SemanticDecoder


def build_s1_model(config: dict[str, object], *, top_k: int = 3) -> Text2SemanticDecoder:
    return Text2SemanticDecoder(config=config, top_k=top_k)


__all__ = ["Text2SemanticDecoder", "build_s1_model"]
