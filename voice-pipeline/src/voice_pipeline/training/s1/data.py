from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class S1Item:
    sample_id: str
    phoneme_ids: torch.Tensor
    semantic_ids: torch.Tensor
    bert_feature: torch.Tensor


class S1Dataset(Dataset[S1Item]):
    def __init__(
        self,
        preprocess_dir: Path,
        *,
        max_sec: int = 57,
        hz: int = 25,
        min_ps_ratio: float = 3.0,
        max_ps_ratio: float = 25.0,
    ) -> None:
        self.preprocess_dir = Path(preprocess_dir)
        self.max_sec = max_sec
        self.hz = hz
        self.min_ps_ratio = min_ps_ratio
        self.max_ps_ratio = max_ps_ratio
        manifest = self.preprocess_dir / "valid_samples.jsonl"
        if not manifest.is_file():
            raise ValueError(f"S1 manifest does not exist: {manifest}")
        sample_ids = self._read_ids(manifest)
        if not sample_ids:
            raise ValueError(f"S1 manifest contains no samples: {manifest}")
        self._items = [self._load(sample_id) for sample_id in sample_ids]
        count = len(self._items)
        repeat_count = max(2, int(100 / count)) if count < 100 else 1
        self._indices = list(range(count)) * repeat_count

    @staticmethod
    def _read_ids(path: Path) -> list[str]:
        sample_ids: list[str] = []
        seen: set[str] = set()
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                sample_id = json.loads(line)["sample_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"invalid S1 manifest line {line_no}: {error}") from error
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or Path(sample_id).name != sample_id
                or "/" in sample_id
                or "\\" in sample_id
                or sample_id in seen
            ):
                raise ValueError(f"invalid or duplicate sample_id at S1 manifest line {line_no}")
            seen.add(sample_id)
            sample_ids.append(sample_id)
        return sample_ids

    @property
    def sample_count(self) -> int:
        return len(self._items)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> S1Item:
        return self._items[self._indices[index]]

    def _load(self, sample_id: str) -> S1Item:
        metadata_path = self.preprocess_dir / "text" / f"{sample_id}.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            phone_ids = metadata["phone_ids"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"{sample_id} has invalid S1 text artifact: {error}") from error
        if (
            metadata.get("sample_id") != sample_id
            or not isinstance(phone_ids, list)
            or not phone_ids
            or any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 732
                for value in phone_ids
            )
        ):
            raise ValueError(f"{sample_id} has invalid S1 phone IDs")

        bert = self._load_tensor(sample_id, "text", f"{sample_id}.bert.pt")
        if (
            bert.ndim != 2
            or bert.shape != (1024, len(phone_ids))
            or not bert.is_floating_point()
            or not torch.isfinite(bert).all()
        ):
            raise ValueError(f"{sample_id} has invalid BERT shape, dtype, or values")

        semantic = self._load_tensor(sample_id, "semantic", f"{sample_id}.pt")
        if semantic.ndim != 1 or semantic.dtype != torch.int64 or semantic.numel() == 0:
            raise ValueError(f"{sample_id} has invalid semantic shape or dtype")
        if semantic.numel() > self.max_sec * self.hz:
            raise ValueError(f"{sample_id} exceeds S1 duration limit")
        if semantic.min().item() < 0 or semantic.max().item() > 1023:
            raise ValueError(f"{sample_id} has invalid semantic token range")
        if len(phone_ids) > self.max_sec * self.hz / 2.5:
            raise ValueError(f"{sample_id} exceeds S1 phone length limit")
        phone_rate = len(phone_ids) / (semantic.numel() / self.hz)
        if not self.min_ps_ratio <= phone_rate <= self.max_ps_ratio:
            raise ValueError(f"{sample_id} has S1 phone/sec outside official bounds")

        return S1Item(
            sample_id,
            torch.tensor(phone_ids, dtype=torch.int64),
            semantic.contiguous(),
            bert.float().contiguous(),
        )

    def _load_tensor(self, sample_id: str, directory: str, filename: str) -> torch.Tensor:
        path = self.preprocess_dir / directory / filename
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"{sample_id} has invalid {directory} tensor: {error}") from error
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{sample_id} has invalid {directory} tensor")
        return value


class S1Collate:
    def __init__(self, pad_value: int = 1024) -> None:
        self.pad_value = pad_value

    def __call__(self, items: list[S1Item]) -> dict[str, torch.Tensor | list[str]]:
        if not items:
            raise ValueError("cannot collate an empty S1 batch")
        phone_lengths = torch.tensor([item.phoneme_ids.numel() for item in items], dtype=torch.int64)
        semantic_lengths = torch.tensor([item.semantic_ids.numel() for item in items], dtype=torch.int64)
        phone_ids = pad_sequence([item.phoneme_ids for item in items], batch_first=True)
        semantic_ids = pad_sequence(
            [item.semantic_ids for item in items], batch_first=True, padding_value=self.pad_value
        )
        bert = torch.zeros(len(items), 1024, int(phone_lengths.max()), dtype=torch.float32)
        for index, item in enumerate(items):
            bert[index, :, : item.bert_feature.shape[1]] = item.bert_feature
        return {
            "sample_ids": [item.sample_id for item in items],
            "phoneme_ids": phone_ids,
            "phoneme_ids_len": phone_lengths,
            "semantic_ids": semantic_ids,
            "semantic_ids_len": semantic_lengths,
            "bert_feature": bert,
        }


__all__ = ["S1Collate", "S1Dataset", "S1Item"]
