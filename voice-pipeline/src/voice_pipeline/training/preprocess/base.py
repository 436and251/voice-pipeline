from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from voice_pipeline.profiles.base import ModelProfile
from voice_pipeline.training.experiment import Experiment


@dataclass(frozen=True, slots=True)
class StageContext:
    experiment: Experiment
    profile: ModelProfile
    config: Mapping[str, object]
    asset_digests: Mapping[str, str]

    @property
    def preprocess_dir(self) -> Path:
        return self.experiment.preprocess_dir


class SampleFailure(Exception):
    def __init__(self, stage: str, category: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.category = category
        self.message = message


@dataclass(frozen=True, slots=True)
class StageSampleResult:
    sample_id: str
    output_paths: list[Path]
    metadata: dict[str, object]
