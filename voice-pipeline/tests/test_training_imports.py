from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_training_config_import_does_not_load_torch(tmp_path: Path):
    source_root = Path(__file__).parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(source_root), environment.get("PYTHONPATH")])
    )
    script = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError(f"configuration import loaded training runtime: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from voice_pipeline.training.config import TrainingConfig
assert TrainingConfig.__name__ == "TrainingConfig"
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
