from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from typer.testing import CliRunner

from voice_pipeline.cli.module import app as module_app
from voice_pipeline.module_api.descriptor import build_descriptor


runner = CliRunner()


def test_descriptor_is_stable_and_side_effect_free(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    payload = build_descriptor()

    assert payload["protocol_version"] == 2
    assert payload["module_id"] == "gpt-sovits-v2proplus"
    assert payload["frameworks"][0]["id"] == "v2ProPlus"
    assert payload["frameworks"][0]["capabilities"] == [
        "preprocess",
        "train",
        "evaluate",
        "listen",
        "promote",
        "infer",
    ]
    assert payload["frameworks"][0]["training_data"] == {
        "kind": "file",
        "extensions": [".list"],
    }
    assert list(tmp_path.iterdir()) == []


def test_descriptor_fields_use_only_supported_types_and_have_three_language_labels():
    fields = build_descriptor()["frameworks"][0]["fields"]

    assert fields
    assert {field["kind"] for field in fields} <= {
        "string",
        "integer",
        "number",
        "boolean",
        "enum",
        "path",
    }
    assert all(set(field["labels"]) == {"zh_CN", "en", "ja"} for field in fields)
    assert all("default" in field and "constraints" in field for field in fields)


def test_module_describe_prints_one_json_object_to_stdout_only():
    result = runner.invoke(module_app, ["describe", "--json"])

    assert result.exit_code == 0
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == build_descriptor()


def test_python_module_describe_does_not_import_torch(tmp_path):
    source_root = Path(__file__).parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(source_root), environment.get("PYTHONPATH")])
    )
    script = """
import builtins
import runpy
import sys

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError(f"descriptor imported heavyweight dependency: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
sys.argv = ["voice-pipeline", "module", "describe", "--json"]
runpy.run_module("voice_pipeline", run_name="__main__")
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
    assert result.stderr == ""
    assert json.loads(result.stdout)["module_id"] == "gpt-sovits-v2proplus"
    assert list(tmp_path.iterdir()) == []
