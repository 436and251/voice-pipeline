import inspect
from pathlib import Path

import pytest
import torch

from voice_pipeline.profiles.v2proplus import V2PROPLUS
from voice_pipeline.training.experiment import Experiment
from voice_pipeline.training.manifest import ManifestItem, ManifestRecord
from voice_pipeline.training.preprocess.base import SampleFailure, StageContext
from voice_pipeline.training.preprocess.semantic_stage import SemanticExtractor, SemanticStage


class FakeSemantic:
    def __init__(self, output):
        self.output = output

    def extract(self, ssl):
        return self.output


def record():
    return ManifestRecord(1, "sample", ManifestItem(Path("voice.wav"), "speaker", "ja", "text"))


def context_with_hubert(tmp_path):
    context = StageContext(
        Experiment.create("run", tmp_path), V2PROPLUS, {}, {"s2g": "base-model-digest"}
    )
    path = context.preprocess_dir / "hubert" / "sample.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.ones(1, 768, 8), path)
    return context


def test_semantic_extractor_has_no_finetuned_checkpoint_override():
    assert list(inspect.signature(SemanticExtractor).parameters) == [
        "base_s2g_path",
        "device",
        "precision",
    ]


def test_semantic_stage_saves_integer_25hz_tokens(tmp_path):
    extractor = FakeSemantic(torch.tensor([[[3, 7, 9]]], dtype=torch.long))
    result = SemanticStage(extractor).run(record(), context_with_hubert(tmp_path))
    saved = torch.load(result.output_paths[0], weights_only=True)
    assert saved.tolist() == [3, 7, 9]
    assert result.metadata == {"shape": [3], "dtype": "torch.int64", "frame_rate": "25hz"}


@pytest.mark.parametrize(
    "source,tokens",
    [
        (torch.full((1, 768, 4), float("nan")), torch.tensor([1])),
        (torch.ones(1, 768, 4), torch.tensor([], dtype=torch.long)),
        (torch.ones(1, 768, 4), torch.tensor([1024], dtype=torch.long)),
        (torch.ones(1, 768, 4), torch.tensor([1.0])),
    ],
)
def test_semantic_stage_rejects_invalid_source_or_tokens(tmp_path, source, tokens):
    context = context_with_hubert(tmp_path)
    torch.save(source, context.preprocess_dir / "hubert" / "sample.pt")
    with pytest.raises(SampleFailure):
        SemanticStage(FakeSemantic(tokens)).run(record(), context)
