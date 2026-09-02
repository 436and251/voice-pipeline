from __future__ import annotations

import json
from pathlib import Path

import torch

from voice_pipeline.common.state import RunState
from voice_pipeline.core.gpt_sovits.features.cnhubert import CNHubertExtractor
from voice_pipeline.core.gpt_sovits.features.speaker import SpeakerEncoder
from voice_pipeline.core.gpt_sovits.frontend.multilingual import MultilingualFrontend
from voice_pipeline.pipeline.graph import StageGraph
from voice_pipeline.training.experiment import Experiment

from .artifacts import atomic_write_text, sha256_file, sha256_tree
from .base import StageContext
from .config import PreprocessConfig
from .hubert_stage import HubertStage
from .pipeline import PreprocessPipeline
from .semantic_stage import SemanticExtractor, SemanticStage
from .sv_stage import SVStage
from .text_stage import TextStage
from .wav32k_stage import Wav32kStage


DEPENDENCIES = {
    "text": set(),
    "wav32k": set(),
    "hubert": {"wav32k"},
    "sv": {"wav32k"},
    "semantic": {"hubert"},
}

STAGE_ASSETS = {
    "text": {"bert", "g2pw", "nltk", "langdetect"},
    "wav32k": set(),
    "hubert": {"hubert"},
    "sv": {"speaker"},
    "semantic": {"s2g"},
}


def build_preprocess_pipeline(
    config: PreprocessConfig,
    selected_stage: str | None = None,
) -> PreprocessPipeline:
    full_graph = StageGraph(DEPENDENCIES)
    selected_names = full_graph.topological_order(selected_stage)
    required_assets = set().union(*(STAGE_ASSETS[name] for name in selected_names))
    asset_paths = _asset_paths(config)
    directory_assets = {"bert", "g2pw", "nltk", "langdetect", "hubert"}
    missing = [
        name
        for name in sorted(required_assets)
        if not (
            asset_paths[name].is_dir()
            if name in directory_assets
            else asset_paths[name].is_file()
        )
    ]
    if missing:
        details = ", ".join(f"{name}={asset_paths[name]}" for name in missing)
        raise FileNotFoundError(f"missing required preprocessing assets: {details}")

    asset_digests = {
        name: sha256_tree(asset_paths[name]) if asset_paths[name].is_dir() else sha256_file(asset_paths[name])
        for name in sorted(required_assets)
    }
    effective_precision = (
        "fp16" if config.precision == "fp16" and torch.device(config.device).type == "cuda" else "fp32"
    )
    stages = {}
    if "text" in selected_names:
        frontend = MultilingualFrontend(
            asset_paths["bert"],
            asset_paths["g2pw"],
            asset_paths["nltk"],
            asset_paths["langdetect"],
            config.device,
        )
        stages["text"] = TextStage(frontend)
    if "wav32k" in selected_names:
        stages["wav32k"] = Wav32kStage()
    if "hubert" in selected_names:
        extractor = CNHubertExtractor(asset_paths["hubert"], config.device, effective_precision)
        stages["hubert"] = HubertStage(extractor, effective_precision)
    if "sv" in selected_names:
        stages["sv"] = SVStage(SpeakerEncoder(asset_paths["speaker"], config.device))
    if "semantic" in selected_names:
        stages["semantic"] = SemanticStage(
            SemanticExtractor(asset_paths["s2g"], config.device, effective_precision)
        )

    experiment = Experiment.create(config.experiment_name, config.output_root)
    atomic_write_text(
        experiment.preprocess_dir / "assets.json",
        json.dumps(
            {name: {"path": str(asset_paths[name]), "sha256": digest} for name, digest in asset_digests.items()},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    )
    graph = StageGraph({name: DEPENDENCIES[name] & stages.keys() for name in stages})
    context = StageContext(
        experiment,
        config.profile,
        {"device": config.device, "precision": effective_precision, "resume": config.resume},
        asset_digests,
    )
    return PreprocessPipeline(
        stages,
        graph,
        RunState(experiment.preprocess_dir / "state.json"),
        context,
    )


def _asset_paths(config: PreprocessConfig) -> dict[str, Path]:
    profile = config.profile
    return {
        "s2g": config.project_root / profile.s2g_relative_path,
        "bert": config.project_root / profile.bert_relative_path,
        "hubert": config.project_root / profile.hubert_relative_path,
        "speaker": config.project_root / profile.speaker_relative_path,
        "g2pw": config.project_root / profile.g2pw_relative_path,
        "nltk": config.project_root / profile.nltk_data_relative_path,
        "langdetect": config.project_root / profile.langdetect_relative_path,
    }
