from pathlib import Path

from voice_pipeline.pipeline.signature import compute_stage_signature


def test_signature_changes_when_manifest_changes(tmp_path: Path):
    manifest = tmp_path / "train.list"
    manifest.write_text("a", encoding="utf-8")
    first = compute_stage_signature("text", {"language": "ja"}, [manifest], "v2ProPlus", "1")
    manifest.write_text("b", encoding="utf-8")
    second = compute_stage_signature("text", {"language": "ja"}, [manifest], "v2ProPlus", "1")
    assert first != second


def test_signature_is_stable_for_equivalent_config(tmp_path: Path):
    manifest = tmp_path / "train.list"
    manifest.write_text("a", encoding="utf-8")
    a = compute_stage_signature("text", {"b": 2, "a": 1}, [manifest], "v2ProPlus", "1")
    b = compute_stage_signature("text", {"a": 1, "b": 2}, [manifest], "v2ProPlus", "1")
    assert a == b
