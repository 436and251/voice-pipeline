from pathlib import Path
import torch
from voice_pipeline.core.gpt_sovits.compatibility.checkpoints import save_sovits, load_sovits


def test_sovits_checkpoint_uses_official_06_header(tmp_path: Path):
    path = tmp_path / "s2.pth"
    payload = {"weight": {"x": torch.tensor([1])}, "config": {"model": {"version": "v2ProPlus"}}}
    save_sovits(path, payload)
    assert path.read_bytes()[:2] == b"06"
    loaded = load_sovits(path)
    assert loaded["config"]["model"]["version"] == "v2ProPlus"
    assert torch.equal(loaded["weight"]["x"], torch.tensor([1]))
