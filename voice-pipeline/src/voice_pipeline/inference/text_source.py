from pathlib import Path


def resolve_text_source(text: str | None, text_file: Path | None) -> str:
    if (text is None) == (text_file is None):
        raise ValueError("provide exactly one of text or text_file")
    if text is not None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("inline text is empty")
        return text.strip()

    path = Path(text_file)
    if path.suffix.lower() != ".txt":
        raise ValueError("text_file must be a .txt file")
    if not path.is_file():
        raise ValueError(f"text_file does not exist: {path}")
    try:
        resolved = path.read_text(encoding="utf-8-sig").strip()
    except UnicodeDecodeError as error:
        raise ValueError(f"text_file must be valid UTF-8: {path}") from error
    except OSError as error:
        raise ValueError(f"cannot read text_file {path}: {error}") from error
    if not resolved:
        raise ValueError("text_file is empty")
    return resolved


__all__ = ["resolve_text_source"]
