from pathlib import Path
import pytest

from voice_pipeline.common.errors import ManifestError
from voice_pipeline.training.manifest import load_manifest


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
