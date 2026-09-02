from __future__ import annotations

import json
from pathlib import Path
import wave

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from voice_pipeline.core.gpt_sovits.s2_v2proplus.mel_processing import spectrogram_torch


S2Item = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class S2Dataset(Dataset[S2Item]):
    def __init__(self, preprocess_dir: Path) -> None:
        self.preprocess_dir = Path(preprocess_dir)
        manifest = self.preprocess_dir / "valid_samples.jsonl"
        if not manifest.is_file():
            raise ValueError(f"S2 manifest does not exist: {manifest}")
        self._sample_ids = self._read_ids(manifest)
        if not self._sample_ids:
            raise ValueError(f"S2 manifest contains no samples: {manifest}")
        self._indices = list(range(len(self._sample_ids))) * max(2, int(100 / len(self._sample_ids)))
        for sample_id in self._sample_ids:
            self._load(sample_id)

    @staticmethod
    def _read_ids(path: Path) -> list[str]:
        sample_ids: list[str] = []
        seen: set[str] = set()
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                sample_id = json.loads(line)["sample_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"invalid S2 manifest line {line_no}: {error}") from error
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or Path(sample_id).name != sample_id
                or "/" in sample_id
                or "\\" in sample_id
                or sample_id in seen
            ):
                raise ValueError(f"invalid or duplicate sample_id at S2 manifest line {line_no}")
            seen.add(sample_id)
            sample_ids.append(sample_id)
        return sample_ids

    @property
    def sample_count(self) -> int:
        return len(self._sample_ids)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> S2Item:
        return self._load(self._sample_ids[self._indices[index]])

    def _load(self, sample_id: str) -> S2Item:
        text = self._load_text(sample_id)
        wav, spec = self._load_wav(sample_id)
        ssl = self._load_tensor(sample_id, "hubert")
        sv = self._load_tensor(sample_id, "sv")
        if ssl.ndim != 3 or ssl.shape[:2] != (1, 768) or not ssl.is_floating_point():
            raise ValueError(f"{sample_id} has invalid hubert shape or dtype")
        if not torch.isfinite(ssl).all():
            raise ValueError(f"{sample_id} has non-finite hubert tensor")
        if ssl.shape[-1] == spec.shape[-1] - 1:
            ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(ssl.dtype)
        elif ssl.shape[-1] != spec.shape[-1]:
            raise ValueError(f"{sample_id} has mismatched hubert and spectrogram frames")
        if sv.ndim != 2 or sv.shape != (1, 20480) or not sv.is_floating_point():
            raise ValueError(f"{sample_id} has invalid sv shape or dtype")
        if not torch.isfinite(sv).all():
            raise ValueError(f"{sample_id} has non-finite sv tensor")
        return ssl.contiguous(), spec, wav, text, sv.contiguous()

    def _load_text(self, sample_id: str) -> torch.Tensor:
        path = self.preprocess_dir / "text" / f"{sample_id}.json"
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            phone_ids = row["phone_ids"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"{sample_id} has invalid text artifact: {error}") from error
        if (
            row.get("sample_id") != sample_id
            or not isinstance(phone_ids, list)
            or not phone_ids
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in phone_ids)
        ):
            raise ValueError(f"{sample_id} has invalid text phone_ids")
        return torch.tensor(phone_ids, dtype=torch.long)

    def _load_wav(self, sample_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.preprocess_dir / "wav32k" / f"{sample_id}.wav"
        try:
            with wave.open(str(path), "rb") as stream:
                channels = stream.getnchannels()
                sample_width = stream.getsampwidth()
                sample_rate = stream.getframerate()
                frame_count = stream.getnframes()
                payload = stream.readframes(frame_count)
        except (OSError, EOFError, wave.Error) as error:
            raise ValueError(f"{sample_id} has invalid wav32k artifact: {error}") from error
        duration = frame_count / sample_rate if sample_rate else 0.0
        if channels != 1 or sample_width != 2 or sample_rate != 32000 or not 0.6 < duration < 54.0:
            raise ValueError(f"{sample_id} has invalid wav32k format or duration")
        samples = torch.frombuffer(bytearray(payload), dtype=torch.int16).to(torch.float32) / 32768.0
        if samples.numel() != frame_count:
            raise ValueError(f"{sample_id} has truncated wav32k payload")
        wav = samples.unsqueeze(0)
        spec = spectrogram_torch(wav, 2048, 32000, 640, 2048, center=False).squeeze(0)
        return wav, spec

    def _load_tensor(self, sample_id: str, directory: str) -> torch.Tensor:
        path = self.preprocess_dir / directory / f"{sample_id}.pt"
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"{sample_id} has invalid {directory} artifact: {error}") from error
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{sample_id} has invalid {directory} artifact")
        return value


class DeterministicEpochSampler(Sampler[int]):
    def __init__(self, dataset: S2Dataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


class S2Collate:
    def __call__(self, batch: list[S2Item]) -> tuple[torch.Tensor, ...]:
        order = sorted(range(len(batch)), key=lambda index: batch[index][1].shape[1], reverse=True)
        rows = [batch[index] for index in order]
        ssl_width = 2 * ((max(row[0].shape[2] for row in rows) // 2) + 1)
        spec_width = 2 * ((max(row[1].shape[1] for row in rows) // 2) + 1)
        wav_width = max(row[2].shape[1] for row in rows)
        text_width = max(row[3].shape[0] for row in rows)
        size = len(rows)

        ssl = torch.zeros(size, 768, ssl_width)
        spec = torch.zeros(size, 1025, spec_width)
        wav = torch.zeros(size, 1, wav_width)
        text = torch.zeros(size, text_width, dtype=torch.long)
        sv = torch.zeros(size, 20480)
        ssl_lengths = torch.empty(size, dtype=torch.long)
        spec_lengths = torch.empty(size, dtype=torch.long)
        wav_lengths = torch.empty(size, dtype=torch.long)
        text_lengths = torch.empty(size, dtype=torch.long)

        for index, (row_ssl, row_spec, row_wav, row_text, row_sv) in enumerate(rows):
            ssl[index, :, : row_ssl.shape[2]] = row_ssl[0]
            spec[index, :, : row_spec.shape[1]] = row_spec
            wav[index, :, : row_wav.shape[1]] = row_wav
            text[index, : row_text.shape[0]] = row_text
            sv[index] = row_sv[0]
            ssl_lengths[index] = row_ssl.shape[2]
            spec_lengths[index] = row_spec.shape[1]
            wav_lengths[index] = row_wav.shape[1]
            text_lengths[index] = row_text.shape[0]
        return ssl, ssl_lengths, spec, spec_lengths, wav, wav_lengths, text, text_lengths, sv
