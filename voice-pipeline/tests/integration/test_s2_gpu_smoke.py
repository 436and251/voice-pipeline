from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import shutil

import pytest
import torch

from voice_pipeline.common.logging import PipelineLogger
from voice_pipeline.training.s2 import S2TrainConfig, S2Trainer, TrainingCursor, checkpoint_path


pytestmark = pytest.mark.gpu


def test_real_v2proplus_s2_runs_one_cuda_step_and_restores(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real S2 smoke test")
    project_root = Path(__file__).parents[2]
    generator = project_root / "models/pretrained/v2proplus/s2/s2Gv2ProPlus.pth"
    discriminator = project_root / "models/pretrained/v2proplus/s2/s2Dv2ProPlus.pth"
    if not generator.is_file() or not discriminator.is_file():
        pytest.skip("local v2ProPlus S2 weights are required")

    preprocess = tmp_path / "preprocess"
    shutil.copytree(project_root / "tests/fixtures/s2_smoke/preprocess", preprocess)
    output = tmp_path / "experiment"
    log_path = output / "training/s2/events.jsonl"
    config = S2TrainConfig(
        preprocess_dir=preprocess,
        output_dir=output,
        base_s2g_path=generator,
        base_s2d_path=discriminator,
        target_optimizer_steps=1,
        checkpoint_every_steps=1,
        batch_size=1,
    )
    trainer = S2Trainer.from_pretrained(config, logger=PipelineLogger(log_path, echo=False))
    assert trainer.train() == TrainingCursor(1, 0, 1)

    checkpoint = checkpoint_path(output, 1)
    assert checkpoint.is_file()
    assert not list(checkpoint.parent.glob("*.tmp"))
    batch_event = next(
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"] == "batch"
    )
    loss_names = ("discriminator_total", "generator_total", "feature", "mel", "kl_ssl", "kl", "learning_rate")
    assert all(math.isfinite(batch_event["metrics"][name]) for name in loss_names)
    gradient_norms = [batch_event["metrics"]["grad_norm_d"], batch_event["metrics"]["grad_norm_g"]]
    if not all(math.isfinite(value) for value in gradient_norms):
        assert trainer.scaler.get_scale() < 65536.0

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    restored = S2Trainer.from_pretrained(config, resume_from=checkpoint)
    assert restored.cursor == TrainingCursor(1, 0, 1)
    del restored
    gc.collect()
    torch.cuda.empty_cache()
