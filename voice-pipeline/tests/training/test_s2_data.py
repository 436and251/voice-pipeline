from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import wave

import pytest
import torch

from voice_pipeline.training.s2.data import DeterministicEpochSampler, S2Collate, S2Dataset


FIXTURE = Path(__file__).parents[1] / "fixtures" / "s2_smoke"


def test_fixed_fixture_checksums() -> None:
    expected = json.loads((FIXTURE / "checksums.json").read_text(encoding="utf-8"))
    actual = {}
    for path in sorted((FIXTURE / "preprocess").rglob("*")):
        if path.is_file():
            payload = path.read_bytes()
            actual[path.relative_to(FIXTURE).as_posix()] = {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
    assert actual == expected


def test_dataset_reads_five_samples_and_applies_official_repetition() -> None:
    dataset = S2Dataset(FIXTURE / "preprocess")
    assert dataset.sample_count == 5
    assert len(dataset) == 100

    ssl, spec, wav, text, sv = dataset[0]
    assert ssl.shape == (1, 768, 50)
    assert torch.equal(ssl[..., -1], ssl[..., -2])
    assert spec.shape == (1025, 50)
    assert wav.shape == (1, 32_000)
    assert text.dtype == torch.int64 and text.numel() > 0
    assert sv.shape == (1, 20_480)


def test_dataset_leaves_already_aligned_hubert_unchanged(tmp_path: Path) -> None:
    root = _copy_preprocess(tmp_path)
    path = root / "hubert" / "s2-smoke-01.pt"
    raw = torch.load(path, map_location="cpu", weights_only=True)
    aligned = torch.cat((raw, raw[..., -1:] + 1), dim=-1)
    torch.save(aligned, path)
    loaded = S2Dataset(root)[0][0]
    assert torch.equal(loaded, aligned)


def test_epoch_sampler_is_repeatable_without_changing_global_rng() -> None:
    dataset = S2Dataset(FIXTURE / "preprocess")
    sampler = DeterministicEpochSampler(dataset, seed=1234)
    before = torch.random.get_rng_state()
    first = list(sampler)
    assert torch.equal(before, torch.random.get_rng_state())
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)
    assert sorted(first) == sorted(list(sampler))


def test_collate_returns_exact_sorted_nine_tensor_batch() -> None:
    long = (
        torch.ones(1, 768, 5),
        torch.full((1025, 5), 2.0),
        torch.ones(1, 9),
        torch.tensor([3, 4, 5]),
        torch.ones(1, 20480),
    )
    short = (
        torch.ones(1, 768, 3),
        torch.ones(1025, 3),
        torch.ones(1, 4),
        torch.tensor([6]),
        torch.full((1, 20480), 2.0),
    )
    batch = S2Collate()([short, long])
    assert len(batch) == 9
    ssl, ssl_lengths, spec, spec_lengths, wav, wav_lengths, text, text_lengths, sv = batch
    assert ssl.shape == (2, 768, 6)
    assert spec.shape == (2, 1025, 6)
    assert wav.shape == (2, 1, 9)
    assert text.dtype == torch.int64 and text.shape == (2, 3)
    assert sv.shape == (2, 20480)
    assert ssl_lengths.tolist() == [5, 3]
    assert spec_lengths.tolist() == [5, 3]
    assert wav_lengths.tolist() == [9, 4]
    assert text_lengths.tolist() == [3, 1]
    assert torch.equal(spec[0, :, :5], long[1])
    assert torch.count_nonzero(ssl[1, :, 3:]) == 0
    assert torch.count_nonzero(spec[1, :, 3:]) == 0
    assert torch.count_nonzero(wav[1, :, 4:]) == 0
    assert torch.count_nonzero(text[1, 1:]) == 0


def _copy_preprocess(tmp_path: Path) -> Path:
    destination = tmp_path / "preprocess"
    shutil.copytree(FIXTURE / "preprocess", destination)
    return destination


def test_dataset_rejects_missing_artifact(tmp_path: Path) -> None:
    root = _copy_preprocess(tmp_path)
    (root / "sv" / "s2-smoke-01.pt").unlink()
    with pytest.raises(ValueError, match=r"s2-smoke-01.*sv"):
        S2Dataset(root)


@pytest.mark.parametrize("phone_ids", [[], [1, "bad"]])
def test_dataset_rejects_bad_phone_ids(tmp_path: Path, phone_ids: list[object]) -> None:
    root = _copy_preprocess(tmp_path)
    path = root / "text" / "s2-smoke-01.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["phone_ids"] = phone_ids
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match=r"s2-smoke-01.*text"):
        S2Dataset(root)


@pytest.mark.parametrize(
    ("directory", "tensor"),
    [
        ("hubert", torch.zeros(1, 767, 49)),
        ("hubert", torch.zeros(1, 768, 47)),
        ("hubert", torch.full((1, 768, 49), float("nan"))),
        ("sv", torch.zeros(1, 20479)),
        ("sv", torch.full((1, 20480), float("inf"))),
    ],
)
def test_dataset_rejects_bad_feature_tensor(tmp_path: Path, directory: str, tensor: torch.Tensor) -> None:
    root = _copy_preprocess(tmp_path)
    torch.save(tensor, root / directory / "s2-smoke-01.pt")
    with pytest.raises(ValueError, match=rf"s2-smoke-01.*{directory}"):
        S2Dataset(root)


@pytest.mark.parametrize(
    ("channels", "sample_width", "sample_rate", "frames"),
    [(2, 2, 32000, 32000), (1, 1, 32000, 32000), (1, 2, 16000, 16000), (1, 2, 32000, 19200), (1, 2, 32000, 1728000)],
)
def test_dataset_rejects_bad_wav(
    tmp_path: Path, channels: int, sample_width: int, sample_rate: int, frames: int
) -> None:
    root = _copy_preprocess(tmp_path)
    path = root / "wav32k" / "s2-smoke-01.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(sample_width)
        stream.setframerate(sample_rate)
        stream.writeframes(bytes(frames * channels * sample_width))
    with pytest.raises(ValueError, match=r"s2-smoke-01.*wav32k"):
        S2Dataset(root)
