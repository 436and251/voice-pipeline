from pathlib import Path
import pytest

from voice_pipeline.common.errors import ManifestError
from voice_pipeline.training.manifest import (
    allowed_bad_records,
    load_manifest,
    read_manifest_records,
)


def test_load_manifest_valid_entry(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    manifest = tmp_path / "train.list"
    manifest.write_text(f"{wav}|speaker|ja|こんにちは。\n", encoding="utf-8")

    items = load_manifest(manifest)

    assert len(items) == 1
    assert items[0].audio_path == wav
    assert items[0].speaker == "speaker"
    assert items[0].language == "ja"
    assert items[0].text == "こんにちは。"


def test_manifest_rejects_missing_audio(tmp_path: Path):
    manifest = tmp_path / "train.list"
    manifest.write_text(f"{tmp_path / 'missing.wav'}|speaker|ja|こんにちは。\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="audio file does not exist"):
        load_manifest(manifest)


def test_manifest_rejects_empty_text(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    manifest = tmp_path / "train.list"
    manifest.write_text(f"{wav}|speaker|ja|\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="text is empty"):
        load_manifest(manifest)


def test_preprocess_manifest_keeps_valid_rows_and_reports_bad_rows(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    manifest = tmp_path / "data.list"
    manifest.write_text(
        "a.wav|speaker|ja|こんにちは。\n"
        "broken|row\n"
        f"{wav}|speaker|ko|안녕\n",
        encoding="utf-8",
    )

    result = read_manifest_records(manifest)

    assert result.total_records == 3
    assert [record.item.language for record in result.records] == ["ja"]
    assert result.records[0].item.audio_path == wav
    assert [issue.category for issue in result.issues] == ["malformed", "unsupported_language"]


@pytest.mark.parametrize(
    ("total", "allowed"),
    [(0, 0), (1, 1), (6, 2), (10, 2), (21, 5), (100, 5)],
)
def test_bad_record_allowance_rounds_up_and_caps_at_five(total: int, allowed: int):
    assert allowed_bad_records(total) == allowed


def test_exact_duplicate_is_reported_without_speaker_count_validation(tmp_path: Path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"RIFF")
    second.write_bytes(b"RIFF")
    duplicate = f"{first}|speaker-a|ja|こんにちは。"
    manifest = tmp_path / "data.list"
    manifest.write_text(
        f"{duplicate}\n{duplicate}\n{second}|speaker-b|ja|さようなら。\n",
        encoding="utf-8",
    )

    result = read_manifest_records(manifest)

    assert len(result.records) == 2
    assert {record.item.speaker for record in result.records} == {"speaker-a", "speaker-b"}
    assert len({record.sample_id for record in result.records}) == 2
    assert [(issue.line_no, issue.category) for issue in result.issues] == [(2, "duplicate")]
