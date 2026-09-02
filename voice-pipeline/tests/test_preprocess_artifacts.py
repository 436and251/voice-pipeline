import json
from pathlib import Path

import torch

from voice_pipeline.training.preprocess.artifacts import (
    atomic_torch_save,
    atomic_write_text,
    sha256_file,
    sha256_tree,
    write_jsonl,
)


def test_atomic_torch_save_leaves_no_temp_file(tmp_path: Path):
    target = tmp_path / "feature.pt"
    atomic_torch_save(target, torch.ones(2, 3))
    assert torch.equal(torch.load(target, weights_only=True), torch.ones(2, 3))
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_text_and_jsonl_replace_existing_content(tmp_path: Path):
    text_path = tmp_path / "state.json"
    atomic_write_text(text_path, "old")
    atomic_write_text(text_path, "new")
    rows_path = tmp_path / "rows.jsonl"
    write_jsonl(rows_path, [{"text": "你好", "id": 1}])
    assert text_path.read_text(encoding="utf-8") == "new"
    assert json.loads(rows_path.read_text(encoding="utf-8")) == {"id": 1, "text": "你好"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_tree_digest_is_order_independent_and_content_sensitive(tmp_path: Path):
    tree = tmp_path / "model"
    tree.mkdir()
    (tree / "b.bin").write_bytes(b"b")
    (tree / "a.json").write_bytes(b"a")
    first = sha256_tree(tree)
    (tree / "b.bin").write_bytes(b"c")
    assert sha256_tree(tree) != first
    assert sha256_file(tree / "a.json") == sha256_file(tree / "a.json")
