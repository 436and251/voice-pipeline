import json
from pathlib import Path

import pytest

from voice_pipeline.common.state import RunState
from voice_pipeline.pipeline.graph import StageGraph
from voice_pipeline.profiles.v2proplus import V2PROPLUS
from voice_pipeline.training.experiment import Experiment
from voice_pipeline.training.manifest import ManifestIssue, ManifestItem, ManifestRecord
from voice_pipeline.training.preprocess.base import SampleFailure, StageContext, StageSampleResult
from voice_pipeline.training.preprocess.pipeline import PreprocessPipeline, QuarantineLimitExceeded


class FakeStage:
    def __init__(self, name, dependencies, calls, *, fail_ids=(), version="1"):
        self.name = name
        self.dependencies = set(dependencies)
        self.calls = calls
        self.fail_ids = set(fail_ids)
        self.version = version

    def signature(self, record, context):
        return f"{record.sample_id}:{self.version}"

    def run(self, record, context):
        self.calls.append((self.name, record.sample_id))
        if record.sample_id in self.fail_ids:
            raise SampleFailure(self.name, "fake_failure", "intentional")
        path = context.preprocess_dir / self.name / f"{record.sample_id}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.version, encoding="utf-8")
        return StageSampleResult(record.sample_id, [path], {"version": self.version})

    def validate_cached(self, record, entry, context):
        return entry["metadata"].get("version") == self.version


def records(count):
    return [
        ManifestRecord(i + 1, f"s{i}", ManifestItem(Path(f"audio-{i}.wav"), "speaker", "ja", f"text {i}"))
        for i in range(count)
    ]


def stages(calls, *, failures=None):
    failures = failures or {}
    return {
        "text": FakeStage("text", set(), calls, fail_ids=failures.get("text", ())),
        "wav32k": FakeStage("wav32k", set(), calls, fail_ids=failures.get("wav32k", ())),
        "hubert": FakeStage("hubert", {"wav32k"}, calls, fail_ids=failures.get("hubert", ())),
        "sv": FakeStage("sv", {"wav32k"}, calls, fail_ids=failures.get("sv", ())),
        "semantic": FakeStage("semantic", {"hubert"}, calls, fail_ids=failures.get("semantic", ())),
    }


def make_pipeline(tmp_path, stage_map):
    experiment = Experiment.create("run", tmp_path)
    context = StageContext(experiment, V2PROPLUS, {}, {})
    graph = StageGraph({name: stage.dependencies for name, stage in stage_map.items()})
    return PreprocessPipeline(stage_map, graph, RunState(experiment.preprocess_dir / "state.json"), context)


def read_index_ids(root):
    found = []
    for path in (root / "run" / "preprocess").glob("*/index.jsonl"):
        found.extend(json.loads(line)["sample_id"] for line in path.read_text(encoding="utf-8").splitlines())
    return found


def test_pipeline_runs_in_dependency_order_and_resumes_completed_samples(tmp_path):
    calls = []
    pipeline = make_pipeline(tmp_path, stages(calls))
    pipeline.run(records(2), [])
    assert [name for name, _ in calls[:5]] == ["text", "text", "wav32k", "wav32k", "hubert"]

    calls.clear()
    pipeline.run(records(2), [])
    assert calls == []


def test_one_stage_failure_quarantines_sample_from_every_stage_index(tmp_path):
    calls = []
    pipeline = make_pipeline(tmp_path, stages(calls, failures={"hubert": {"s5"}}))
    summary = pipeline.run(records(6), [])

    assert summary.valid_sample_ids == [f"s{i}" for i in range(5)]
    assert summary.quarantined[0].sample_id == "s5"
    assert "s5" not in read_index_ids(tmp_path)


def test_pipeline_fails_on_third_bad_record_out_of_ten(tmp_path):
    pipeline = make_pipeline(tmp_path, stages([], failures={"text": {"s0", "s1", "s2"}}))
    with pytest.raises(QuarantineLimitExceeded, match="3 > 2"):
        pipeline.run(records(10), [])
    assert not (tmp_path / "run" / "preprocess" / "valid_samples.jsonl").exists()


def test_six_records_allow_two_bad_but_zero_valid_fails(tmp_path):
    pipeline = make_pipeline(tmp_path, stages([], failures={"text": {"s0", "s1"}}))
    assert pipeline.run(records(6), []).allowed_bad == 2

    empty_pipeline = make_pipeline(tmp_path / "empty", stages([], failures={"text": {"s0"}}))
    with pytest.raises(QuarantineLimitExceeded, match="no valid"):
        empty_pipeline.run(records(1), [])


def test_manifest_issues_consume_the_same_global_allowance(tmp_path):
    issue = ManifestIssue(7, "broken", "malformed", "expected four fields")
    pipeline = make_pipeline(tmp_path, stages([], failures={"text": {"s0", "s1"}}))
    with pytest.raises(QuarantineLimitExceeded, match="3 > 2"):
        pipeline.run(records(7), [issue])


def test_changed_stage_signature_invalidates_only_downstream_samples(tmp_path):
    calls = []
    stage_map = stages(calls)
    pipeline = make_pipeline(tmp_path, stage_map)
    pipeline.run(records(1), [])

    calls.clear()
    stage_map["wav32k"].version = "2"
    pipeline.run(records(1), [])
    assert [name for name, _ in calls] == ["wav32k", "hubert", "sv", "semantic"]


def test_temporary_output_never_creates_a_cache_hit(tmp_path):
    calls = []
    pipeline = make_pipeline(tmp_path, stages(calls))
    pipeline.run(records(1), [])
    index = tmp_path / "run" / "preprocess" / "text" / "index.jsonl"
    row = json.loads(index.read_text(encoding="utf-8"))
    temporary = index.parent / "abandoned.tmp"
    temporary.write_text("partial", encoding="utf-8")
    row["output_paths"] = [str(temporary)]
    index.write_text(json.dumps(row) + "\n", encoding="utf-8")

    calls.clear()
    pipeline.run(records(1), [])
    assert ("text", "s0") in calls


def test_target_stage_runs_only_its_dependency_closure(tmp_path):
    calls = []
    pipeline = make_pipeline(tmp_path, stages(calls))
    pipeline.run(records(1), [], selected_stage="semantic")
    assert [name for name, _ in calls] == ["wav32k", "hubert", "semantic"]


def test_changed_manifest_prunes_stale_samples_from_stage_indexes(tmp_path):
    pipeline = make_pipeline(tmp_path, stages([]))
    pipeline.run(records(2), [])
    pipeline.run(records(1), [])
    assert "s1" not in read_index_ids(tmp_path)
