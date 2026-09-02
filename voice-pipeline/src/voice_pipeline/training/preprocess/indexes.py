from __future__ import annotations

import json
from pathlib import Path

import torch

from voice_pipeline.training.manifest import ManifestRecord

from .artifacts import atomic_write_text


def publish_training_indexes(
    preprocess_dir: Path,
    records: list[ManifestRecord],
    valid_ids: set[str],
) -> list[Path]:
    records_by_id = {record.sample_id: record for record in records}
    missing_records = valid_ids - records_by_id.keys()
    if missing_records:
        raise ValueError(f"valid sample IDs missing from manifest: {', '.join(sorted(missing_records))}")

    text_rows: list[str] = []
    semantic_rows = ["item_name\tsemantic_audio"]
    for record in sorted(
        (records_by_id[sample_id] for sample_id in valid_ids),
        key=lambda item: item.line_no,
    ):
        sample_id = record.sample_id
        text_path = preprocess_dir / "text" / f"{sample_id}.json"
        semantic_path = preprocess_dir / "semantic" / f"{sample_id}.pt"
        try:
            text = json.loads(text_path.read_text(encoding="utf-8"))
            semantic = torch.load(semantic_path, map_location="cpu", weights_only=True)
        except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid preprocessing artifacts for {sample_id}: {error}") from error
        if text.get("sample_id") != sample_id:
            raise ValueError(f"text artifact sample mismatch for {sample_id}")
        phones = text.get("phones")
        phone_ids = text.get("phone_ids")
        normalized = text.get("normalized_text")
        word2ph = text.get("word2ph")
        if (
            not isinstance(phones, list)
            or not phones
            or not isinstance(phone_ids, list)
            or len(phones) != len(phone_ids)
            or not isinstance(normalized, str)
            or "\t" in normalized
            or "\n" in normalized
            or (word2ph is not None and not isinstance(word2ph, list))
        ):
            raise ValueError(f"invalid text artifact for {sample_id}")
        if (
            not isinstance(semantic, torch.Tensor)
            or semantic.dtype != torch.int64
            or semantic.ndim != 1
            or semantic.numel() == 0
            or semantic.min().item() < 0
            or semantic.max().item() > 1023
        ):
            raise ValueError(f"invalid semantic artifact for {sample_id}")

        word2ph_text = "" if word2ph is None else " ".join(str(value) for value in word2ph)
        text_rows.append(f"{sample_id}\t{' '.join(phones)}\t{word2ph_text}\t{normalized}")
        semantic_rows.append(f"{sample_id}\t{' '.join(str(value) for value in semantic.tolist())}")

    text_path = preprocess_dir / "2-name2text.txt"
    semantic_path = preprocess_dir / "6-name2semantic.tsv"
    atomic_write_text(text_path, "\n".join(text_rows) + "\n")
    atomic_write_text(semantic_path, "\n".join(semantic_rows) + "\n")
    return [text_path, semantic_path]
