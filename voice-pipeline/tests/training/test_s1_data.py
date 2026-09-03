from __future__ import annotations

from pathlib import Path
import shutil

import pytest
import torch

from voice_pipeline.training.s1.data import S1Collate, S1Dataset, S1Item
from voice_pipeline.training.sampler import DeterministicEpochSampler


FIXTURE = Path(__file__).parents[1] / "fixtures" / "s2_smoke" / "preprocess"


def test_dataset_reads_fixed_samples_and_applies_official_repetition() -> None:
    dataset = S1Dataset(FIXTURE)

    assert dataset.sample_count == 5
    assert len(dataset) == 100
    item = dataset[0]
    assert item.sample_id == "s2-smoke-01"
    assert item.phoneme_ids.dtype == torch.int64 and item.phoneme_ids.shape == (8,)
    assert item.semantic_ids.dtype == torch.int64 and item.semantic_ids.shape == (25,)
    assert item.bert_feature.dtype == torch.float32 and item.bert_feature.shape == (1024, 8)


def test_collate_uses_official_padding_values() -> None:
    short = S1Item(
        "short",
        torch.tensor([1, 2]),
        torch.tensor([3, 4, 5]),
        torch.ones(1024, 2),
    )
    long = S1Item(
        "long",
        torch.tensor([6, 7, 8]),
        torch.tensor([9, 10, 11, 12]),
        torch.full((1024, 3), 2.0),
    )

    batch = S1Collate()([short, long])

    assert batch["sample_ids"] == ["short", "long"]
    assert batch["phoneme_ids"].tolist() == [[1, 2, 0], [6, 7, 8]]
    assert batch["phoneme_ids_len"].tolist() == [2, 3]
    assert batch["semantic_ids"].tolist() == [[3, 4, 5, 1024], [9, 10, 11, 12]]
    assert batch["semantic_ids_len"].tolist() == [3, 4]
    assert batch["bert_feature"].shape == (2, 1024, 3)
    assert torch.equal(batch["bert_feature"][0, :, 2], torch.zeros(1024))


@pytest.mark.parametrize(
    ("relative_path", "replacement", "message"),
    [
        ("text/s2-smoke-01.bert.pt", torch.zeros(512, 8), "s2-smoke-01 has invalid BERT"),
        ("semantic/s2-smoke-01.pt", torch.tensor([1024]), "s2-smoke-01 has invalid semantic"),
        ("semantic/s2-smoke-01.pt", torch.arange(57 * 25 + 1), "s2-smoke-01 exceeds S1 duration"),
    ],
)
def test_dataset_rejects_invalid_canonical_artifacts(
    tmp_path: Path,
    relative_path: str,
    replacement: torch.Tensor,
    message: str,
) -> None:
    root = tmp_path / "preprocess"
    shutil.copytree(FIXTURE, root)
    torch.save(replacement, root / relative_path)

    with pytest.raises(ValueError, match=message):
        S1Dataset(root)


def test_dataset_rejects_phone_rate_outside_official_bounds(tmp_path: Path) -> None:
    root = tmp_path / "preprocess"
    shutil.copytree(FIXTURE, root)
    torch.save(torch.arange(100, dtype=torch.int64), root / "semantic" / "s2-smoke-01.pt")

    with pytest.raises(ValueError, match="s2-smoke-01 has S1 phone/sec outside"):
        S1Dataset(root)


def test_epoch_sampler_is_deterministic_without_consuming_global_rng() -> None:
    dataset = S1Dataset(FIXTURE)
    sampler = DeterministicEpochSampler(dataset, seed=1234)
    before = torch.random.get_rng_state()

    first = list(sampler)
    assert torch.equal(before, torch.random.get_rng_state())
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)
    assert sorted(first) == sorted(list(sampler))
