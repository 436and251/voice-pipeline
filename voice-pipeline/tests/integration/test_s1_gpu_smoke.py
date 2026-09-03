from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import shutil

import pytest
import torch

from voice_pipeline.common.logging import PipelineLogger
from voice_pipeline.training.s1 import S1TrainConfig, S1Trainer, S1TrainingCursor, checkpoint_path


pytestmark = pytest.mark.gpu


def test_real_v2proplus_s1_runs_four_cuda_minibatches_and_restores(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real S1 smoke test")
    project_root = Path(__file__).parents[2]
    base_s1 = project_root / "models/pretrained/v2proplus/s1/s1v3.ckpt"
    if not base_s1.is_file():
        pytest.skip("local v2ProPlus S1 weights are required")
    preprocess = tmp_path / "preprocess"
    shutil.copytree(project_root / "tests/fixtures/s2_smoke/preprocess", preprocess)
    output = tmp_path / "experiment"
    log_path = output / "training/s1/events.jsonl"
    config = S1TrainConfig(
        preprocess_dir=preprocess,
        output_dir=output,
        base_s1_path=base_s1,
        target_optimizer_steps=1,
        checkpoint_every_steps=1,
        batch_size=1,
    )
    trainer = None
    restored = None
    try:
        trainer = S1Trainer.from_pretrained(config, logger=PipelineLogger(log_path, echo=False))
        initial_scale = trainer.scaler.get_scale()
        assert initial_scale == 65536.0
        assert trainer.train() == S1TrainingCursor(1, 0, 4, 0)
        checkpoint = checkpoint_path(output, 1)
        assert checkpoint.is_file()
        assert not list(checkpoint.parent.glob("*.tmp"))
        events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        mini_batches = [event for event in events if event["event"] == "mini_batch"]
        optimizer_events = [event for event in events if event["event"] == "optimizer"]
        assert len(mini_batches) == 4 and len(optimizer_events) == 1
        assert all(math.isfinite(event["metrics"]["loss"]) for event in mini_batches)
        assert all(math.isfinite(event["metrics"]["top3_accuracy"]) for event in mini_batches)
        assert optimizer_events[0]["metrics"]["learning_rate"] == 0.01
        assert optimizer_events[0]["metrics"]["scaler_scale"] <= initial_scale
        restored = S1Trainer.from_pretrained(config, resume_from=checkpoint)
        assert restored.cursor == S1TrainingCursor(1, 0, 4, 0)
    finally:
        del restored, trainer
        gc.collect()
        torch.cuda.empty_cache()
        if output.exists():
            shutil.rmtree(output)
        assert not output.exists()
