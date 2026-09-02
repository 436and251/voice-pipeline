from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from voice_pipeline.core.gpt_sovits.compatibility.s2_checkpoint import load_s2_generator
from voice_pipeline.training.manifest import ManifestRecord

from .artifacts import atomic_torch_save, sha256_file
from .base import SampleFailure, StageContext, StageSampleResult


class SemanticExtractor:
    def __init__(self, base_s2g_path: Path, device="cpu", precision="fp32"):
        if precision not in {"fp16", "fp32"}:
            raise ValueError(f"unsupported semantic precision: {precision}")
        self.model = load_s2_generator(base_s2g_path, device=device).eval()
        if precision == "fp16" and torch.device(device).type == "cuda":
            self.model.half()

    def extract(self, ssl: torch.Tensor) -> torch.Tensor:
        parameter = next(self.model.parameters())
        with torch.inference_mode():
            codes = self.model.extract_latent(ssl.to(device=parameter.device, dtype=parameter.dtype))
        return codes[0, 0].to(device="cpu", dtype=torch.long)


class SemanticStage:
    name = "semantic"
    dependencies = {"hubert"}

    def __init__(self, extractor):
        self.extractor = extractor

    def _hubert_path(self, record: ManifestRecord, context: StageContext) -> Path:
        return context.preprocess_dir / "hubert" / f"{record.sample_id}.pt"

    def signature(self, record: ManifestRecord, context: StageContext) -> str:
        payload = {
            "stage": self.name,
            "version": 1,
            "hubert_sha256": sha256_file(self._hubert_path(record, context)),
            "base_s2g_sha256": context.asset_digests.get("s2g"),
            "frame_rate": context.profile.semantic_frame_rate,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()

    def run(self, record: ManifestRecord, context: StageContext) -> StageSampleResult:
        if context.profile.semantic_frame_rate != "25hz":
            raise ValueError(
                f"profile {context.profile.name} has unsupported semantic frame rate: "
                f"{context.profile.semantic_frame_rate}"
            )
        try:
            ssl = torch.load(self._hubert_path(record, context), map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise SampleFailure(self.name, "invalid_hubert", str(error)) from error
        if (
            not isinstance(ssl, torch.Tensor)
            or ssl.ndim != 3
            or ssl.shape[0] != 1
            or ssl.shape[1] != 768
            or ssl.shape[2] < 1
            or not torch.isfinite(ssl).all()
        ):
            raise SampleFailure(self.name, "invalid_hubert", "invalid or non-finite HuBERT tensor")

        try:
            tokens = self.extractor.extract(ssl)
        except RuntimeError as error:
            raise SampleFailure(self.name, "semantic_extraction", str(error)) from error
        if isinstance(tokens, torch.Tensor) and tokens.ndim == 3 and tokens.shape[:2] == (1, 1):
            tokens = tokens[0, 0]
        integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1 or tokens.numel() == 0:
            raise SampleFailure(self.name, "invalid_tokens", "semantic tokens must be a non-empty rank-one tensor")
        if tokens.dtype not in integer_dtypes:
            raise SampleFailure(self.name, "invalid_tokens", "semantic tokens must use an integer dtype")
        if tokens.min().item() < 0 or tokens.max().item() > 1023:
            raise SampleFailure(self.name, "invalid_tokens", "semantic token IDs must be within 0..1023")

        tokens = tokens.detach().to(device="cpu", dtype=torch.long).contiguous()
        output_path = context.preprocess_dir / self.name / f"{record.sample_id}.pt"
        atomic_torch_save(output_path, tokens)
        metadata = {"shape": list(tokens.shape), "dtype": str(tokens.dtype), "frame_rate": "25hz"}
        return StageSampleResult(record.sample_id, [output_path], metadata)

    def validate_cached(self, record, entry, context) -> bool:
        output_paths = entry.get("output_paths")
        metadata = entry.get("metadata")
        if not isinstance(output_paths, list) or len(output_paths) != 1 or not isinstance(metadata, dict):
            return False
        try:
            tokens = torch.load(Path(output_paths[0]), map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        return (
            isinstance(tokens, torch.Tensor)
            and tokens.dtype == torch.int64
            and tokens.ndim == 1
            and tokens.numel() > 0
            and tokens.min().item() >= 0
            and tokens.max().item() <= 1023
            and metadata == {"shape": list(tokens.shape), "dtype": "torch.int64", "frame_rate": "25hz"}
        )
