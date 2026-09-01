from pathlib import Path
import io
import torch

_HEADER = b"06"


def save_sovits(path: str | Path, payload: dict) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    Path(path).write_bytes(_HEADER + buffer.getvalue())


def load_sovits(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    if raw[:2] != _HEADER:
        raise ValueError("invalid SoVITS checkpoint header")
    return torch.load(io.BytesIO(raw[2:]), map_location="cpu", weights_only=False)
