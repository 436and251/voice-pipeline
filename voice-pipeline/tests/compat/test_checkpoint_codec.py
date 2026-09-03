import io
from pathlib import Path

import torch

from voice_pipeline.core.gpt_sovits.compatibility.checkpoints import save_sovits, load_sovits


def test_sovits_checkpoint_uses_official_06_header(tmp_path: Path):
    path = tmp_path / "s2.pth"
    payload = {"weight": {"x": torch.tensor([1])}, "config": {"model": {"version": "v2ProPlus"}}}
    save_sovits(path, payload)
    raw = path.read_bytes()
    assert raw[:2] == b"06"
    assert raw[2:4] != b"PK"
    official_loaded = torch.load(io.BytesIO(b"PK" + raw[2:]), map_location="cpu", weights_only=False)
    assert torch.equal(official_loaded["weight"]["x"], torch.tensor([1]))
    loaded = load_sovits(path)
    assert loaded["config"]["model"]["version"] == "v2ProPlus"
    assert torch.equal(loaded["weight"]["x"], torch.tensor([1]))


def test_sovits_loader_accepts_raw_official_pretrained_checkpoint(tmp_path: Path):
    path = tmp_path / "pretrained.pth"
    payload = {"weight": {"x": torch.tensor([2])}, "config": {"model": {"version": "v2ProPlus"}}}
    torch.save(payload, path)

    assert path.read_bytes()[:2] == b"PK"
    loaded = load_sovits(path)
    assert torch.equal(loaded["weight"]["x"], torch.tensor([2]))
