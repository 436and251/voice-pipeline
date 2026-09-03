from pathlib import Path
import io
import torch

_HEADER = b"06"


def save_sovits(path: str | Path, payload: dict) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    if raw[:2] != b"PK":
        raise ValueError("unsupported torch checkpoint archive")
    Path(path).write_bytes(_HEADER + raw[2:])


def load_sovits(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    if raw[:2] == _HEADER:
        raw = b"PK" + raw[2:]
    elif raw[:2] != b"PK":
        raise ValueError("invalid SoVITS checkpoint header")
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
