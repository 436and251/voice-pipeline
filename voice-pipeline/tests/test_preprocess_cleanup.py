import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from voice_pipeline.training.preprocess.cleanup import cleanup_after_training


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_cleanup_fixture(tmp_path):
    root = tmp_path / "preprocess"
    hubert = root / "hubert"
    hubert.mkdir(parents=True)
    good = hubert / "good.pt"
    bad = hubert / "bad.pt"
    temporary = root / "text" / ".partial.123.tmp"
    temporary.parent.mkdir(parents=True)
    good.write_bytes(b"good")
    bad.write_bytes(b"bad")
    temporary.write_bytes(b"partial")
    write_jsonl(root / "valid_samples.jsonl", [{"sample_id": "good"}])
    write_jsonl(root / "quarantine.jsonl", [{"sample_id": "bad"}])
    (root / "hubert" / "index.jsonl").write_text("index", encoding="utf-8")
    return SimpleNamespace(
        preprocess_dir=root,
        good_hubert=good,
        bad_hubert=bad,
        temp=temporary,
        index=root / "hubert" / "index.jsonl",
    )


def test_success_cleanup_removes_temp_and_quarantined_but_keeps_valid(tmp_path):
    layout = make_cleanup_fixture(tmp_path)
    report = cleanup_after_training(layout.preprocess_dir, training_succeeded=True)
    assert not layout.temp.exists()
    assert not layout.bad_hubert.exists()
    assert layout.good_hubert.exists()
    assert layout.index.exists()
    assert report.removed_temporary == 1
    assert report.removed_quarantined == 1
    assert report.retained_valid == 1


def test_failed_training_removes_nothing(tmp_path):
    layout = make_cleanup_fixture(tmp_path)
    report = cleanup_after_training(layout.preprocess_dir, training_succeeded=False)
    assert layout.temp.exists() and layout.bad_hubert.exists() and layout.good_hubert.exists()
    assert report.removed_temporary == report.removed_quarantined == 0


def test_cleanup_rejects_output_path_outside_preprocess_root(tmp_path):
    root = tmp_path / "preprocess"
    root.mkdir()
    write_jsonl(root / "valid_samples.jsonl", [{"sample_id": "good"}])
    write_jsonl(root / "quarantine.jsonl", [{"sample_id": "..\\outside"}])
    with pytest.raises(ValueError, match="outside preprocess root"):
        cleanup_after_training(root, training_succeeded=True)
