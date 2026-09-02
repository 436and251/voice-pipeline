import json
from pathlib import Path

import pytest
import torch

from voice_pipeline.training.manifest import ManifestItem, ManifestRecord
from voice_pipeline.training.preprocess.indexes import publish_training_indexes


def records():
    return [
        ManifestRecord(i + 1, name, ManifestItem(Path(f"{name}.wav"), "speaker", "ja", name))
        for i, name in enumerate(["a", "b", "c"])
    ]


def write_artifacts(root, sample_ids):
    for sample_id in sample_ids:
        text = root / "text" / f"{sample_id}.json"
        semantic = root / "semantic" / f"{sample_id}.pt"
        text.parent.mkdir(parents=True, exist_ok=True)
        semantic.parent.mkdir(parents=True, exist_ok=True)
        text.write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "normalized_text": sample_id,
                    "phones": ["a", "."],
                    "phone_ids": [1, 2],
                    "word2ph": None,
                    "bert_shape": [1024, 2],
                }
            ),
            encoding="utf-8",
        )
        torch.save(torch.tensor([3, 7, 9]), semantic)


def test_indexes_share_exact_valid_sample_membership(tmp_path):
    write_artifacts(tmp_path, {"a", "c"})
    outputs = publish_training_indexes(tmp_path, records(), {"a", "c"})

    text_ids = {line.split("\t", 1)[0] for line in outputs[0].read_text(encoding="utf-8").splitlines()}
    semantic_lines = outputs[1].read_text(encoding="utf-8").splitlines()[1:]
    semantic_ids = {line.split("\t", 1)[0] for line in semantic_lines}
    assert text_ids == semantic_ids == {"a", "c"}
    assert outputs[1].read_text(encoding="utf-8").splitlines()[0] == "item_name\tsemantic_audio"


def test_indexes_refuse_missing_or_mismatched_valid_artifacts(tmp_path):
    write_artifacts(tmp_path, {"a"})
    with pytest.raises(ValueError, match="c"):
        publish_training_indexes(tmp_path, records(), {"a", "c"})
    assert not (tmp_path / "2-name2text.txt").exists()
    assert not (tmp_path / "6-name2semantic.tsv").exists()
