from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
import torch


def _asset(name: str, *, child: str | None = None) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} for real preprocessing integration")
    path = Path(value)
    if child is not None:
        path /= child
    if not path.exists():
        pytest.fail(f"missing real preprocessing asset: {path}")
    return path


def test_real_v2proplus_preprocessing_produces_aligned_training_views(tmp_path):
    from voice_pipeline.common.state import RunState
    from voice_pipeline.core.gpt_sovits.features.cnhubert import CNHubertExtractor
    from voice_pipeline.core.gpt_sovits.features.speaker import SpeakerEncoder
    from voice_pipeline.core.gpt_sovits.frontend.multilingual import MultilingualFrontend
    from voice_pipeline.pipeline.graph import StageGraph
    from voice_pipeline.profiles.v2proplus import V2PROPLUS
    from voice_pipeline.training.experiment import Experiment
    from voice_pipeline.training.manifest import read_manifest_records
    from voice_pipeline.training.preprocess.base import StageContext
    from voice_pipeline.training.preprocess.hubert_stage import HubertStage
    from voice_pipeline.training.preprocess.indexes import publish_training_indexes
    from voice_pipeline.training.preprocess.pipeline import PreprocessPipeline
    from voice_pipeline.training.preprocess.semantic_stage import SemanticExtractor, SemanticStage
    from voice_pipeline.training.preprocess.sv_stage import SVStage
    from voice_pipeline.training.preprocess.text_stage import TextStage
    from voice_pipeline.training.preprocess.wav32k_stage import Wav32kStage

    bert = _asset("VOICE_PIPELINE_TEST_BERT_DIR")
    g2pw = _asset("VOICE_PIPELINE_TEST_G2PW_DIR")
    nltk = _asset("VOICE_PIPELINE_TEST_NLTK_DATA")
    langdetect = _asset("VOICE_PIPELINE_TEST_LANGDETECT_DIR")
    hubert = _asset("VOICE_PIPELINE_TEST_HUBERT_DIR")
    speaker = _asset("VOICE_PIPELINE_TEST_SV_CHECKPOINT")
    s2g = _asset("VOICE_PIPELINE_TEST_S2_DIR", child="s2Gv2ProPlus.pth")

    time = np.arange(32_000, dtype=np.float32) / 32_000
    waveform = (np.sin(2 * math.pi * 220 * time) * 8_000).astype(np.int16)
    audio_path = tmp_path / "voice.wav"
    wavfile.write(audio_path, 32_000, waveform)
    manifest_path = tmp_path / "data.list"
    manifest_path.write_text(f"{audio_path}|speaker|ja|こんにちは。\n", encoding="utf-8")
    manifest = read_manifest_records(manifest_path)

    experiment = Experiment.create("real", tmp_path / "runs")
    context = StageContext(
        experiment,
        V2PROPLUS,
        {"resume": True},
        {
            "bert": "real",
            "g2pw": "real",
            "nltk": "real",
            "langdetect": "real",
            "hubert": "real",
            "speaker": "real",
            "s2g": "real",
        },
    )
    stages = {
        "text": TextStage(MultilingualFrontend(bert, g2pw, nltk, langdetect, "cpu")),
        "wav32k": Wav32kStage(),
        "hubert": HubertStage(CNHubertExtractor(hubert, "cpu", "fp32"), "fp32"),
        "sv": SVStage(SpeakerEncoder(speaker, "cpu")),
        "semantic": SemanticStage(SemanticExtractor(s2g, "cpu", "fp32")),
    }
    graph = StageGraph({name: stage.dependencies for name, stage in stages.items()})
    pipeline = PreprocessPipeline(
        stages,
        graph,
        RunState(experiment.preprocess_dir / "state.json"),
        context,
    )
    summary = pipeline.run(manifest.records, manifest.issues)
    indexes = publish_training_indexes(
        experiment.preprocess_dir,
        manifest.records,
        set(summary.valid_sample_ids),
    )

    sample_id = summary.valid_sample_ids[0]
    text_metadata = json.loads(
        (experiment.preprocess_dir / "text" / f"{sample_id}.json").read_text(encoding="utf-8")
    )
    text_bert = torch.load(
        experiment.preprocess_dir / "text" / f"{sample_id}.bert.pt", weights_only=True
    )
    content = torch.load(experiment.preprocess_dir / "hubert" / f"{sample_id}.pt", weights_only=True)
    sv = torch.load(experiment.preprocess_dir / "sv" / f"{sample_id}.pt", weights_only=True)
    semantic = torch.load(
        experiment.preprocess_dir / "semantic" / f"{sample_id}.pt", weights_only=True
    )
    text_ids = {line.split("\t", 1)[0] for line in indexes[0].read_text(encoding="utf-8").splitlines()}
    semantic_ids = {
        line.split("\t", 1)[0]
        for line in indexes[1].read_text(encoding="utf-8").splitlines()[1:]
    }

    assert summary.quarantined == []
    assert len(summary.valid_sample_ids) == 1
    assert text_bert.shape == (1024, len(text_metadata["phone_ids"]))
    assert content.shape[0:2] == (1, 768)
    assert sv.shape == (1, 20_480)
    assert semantic.dtype == torch.int64
    assert text_ids == semantic_ids == {sample_id}
