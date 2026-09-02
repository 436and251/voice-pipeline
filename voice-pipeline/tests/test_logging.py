from datetime import datetime
import json

import pytest

from voice_pipeline.common.logging import PipelineLogger


def test_pipeline_logger_writes_fixed_jsonl_record(tmp_path):
    path = tmp_path / "logs" / "s1.jsonl"
    logger = PipelineLogger(path, echo=False)

    logger.log(
        stage="s1",
        event="optimizer_step",
        mini_step=101,
        optimizer_step=25,
        metrics={"loss": 2.841, "accuracy": 0.742},
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert set(record) == {
        "timestamp",
        "stage",
        "event",
        "mini_step",
        "optimizer_step",
        "metrics",
    }
    assert record["stage"] == "s1"
    assert record["event"] == "optimizer_step"
    assert record["mini_step"] == 101
    assert record["optimizer_step"] == 25
    assert record["metrics"] == {"loss": 2.841, "accuracy": 0.742}
    assert datetime.fromisoformat(record["timestamp"]).tzinfo is not None


def test_pipeline_logger_appends_and_echoes_readable_lines(tmp_path, capsys):
    path = tmp_path / "s2.jsonl"
    logger = PipelineLogger(path)

    logger.log("s2", "batch", mini_step=7, optimizer_step=6, metrics={"loss_g": 1.5})
    logger.log("s2", "checkpoint", optimizer_step=6)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["batch", "checkpoint"]
    output = capsys.readouterr().out
    assert "[S2]" in output
    assert "mini_step=7" in output
    assert "optimizer_step=6" in output
    assert "loss_g=1.5" in output


def test_pipeline_logger_does_not_append_partial_invalid_json(tmp_path):
    path = tmp_path / "training.jsonl"
    logger = PipelineLogger(path, echo=False)
    logger.log("s1", "start")

    with pytest.raises(TypeError):
        logger.log("s1", "bad_metric", metrics={"value": object()})

    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
