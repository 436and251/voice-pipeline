from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol

from voice_pipeline.common.state import RunState, StageStatus
from voice_pipeline.pipeline.graph import StageGraph
from voice_pipeline.training.manifest import ManifestIssue, ManifestRecord, allowed_bad_records

from .artifacts import write_jsonl
from .base import SampleFailure, StageContext, StageSampleResult


class QuarantineLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class QuarantineEntry:
    key: str
    line_no: int
    sample_id: str | None
    audio_path: str | None
    stage: str
    category: str
    message: str


@dataclass(frozen=True, slots=True)
class PreprocessSummary:
    valid_sample_ids: list[str]
    quarantined: list[QuarantineEntry]
    allowed_bad: int
    total_records: int


class PreprocessStage(Protocol):
    name: str
    dependencies: set[str]

    def signature(self, record: ManifestRecord, context: StageContext) -> str: ...

    def run(self, record: ManifestRecord, context: StageContext) -> StageSampleResult: ...

    def validate_cached(self, record: ManifestRecord, entry: dict[str, object], context: StageContext) -> bool: ...


class PreprocessPipeline:
    def __init__(
        self,
        stages: dict[str, PreprocessStage],
        graph: StageGraph,
        state: RunState,
        context: StageContext,
    ):
        if set(stages) != set(graph.dependencies):
            raise ValueError("stage map and dependency graph must contain the same stages")
        self.stages = stages
        self.graph = graph
        self.state = state
        self.context = context

    def run(
        self,
        records: list[ManifestRecord],
        initial_issues: list[ManifestIssue],
        selected_stage: str | None = None,
    ) -> PreprocessSummary:
        total_records = len(records) + len(initial_issues)
        allowed_bad = allowed_bad_records(total_records)
        quarantine = {
            f"line-{issue.line_no}": QuarantineEntry(
                key=f"line-{issue.line_no}",
                line_no=issue.line_no,
                sample_id=None,
                audio_path=None,
                stage="manifest",
                category=issue.category,
                message=issue.message,
            )
            for issue in initial_issues
        }
        indexes = {name: self._read_index(name) for name in self.stages}
        quarantine_path = self.context.preprocess_dir / "quarantine.jsonl"
        valid_path = self.context.preprocess_dir / "valid_samples.jsonl"
        if selected_stage is None:
            valid_path.unlink(missing_ok=True)
        self._write_quarantine(quarantine_path, quarantine)
        self._check_limit(quarantine, allowed_bad)

        for stage_name in self.graph.topological_order(selected_stage):
            stage = self.stages[stage_name]
            changed = False
            started = False
            warnings = 0
            for record in records:
                if record.sample_id in quarantine:
                    continue
                signature = stage.signature(record, self.context)
                cached = indexes[stage_name].get(record.sample_id)
                if cached is not None and self._cache_is_valid(stage, record, cached, signature):
                    continue

                changed = True
                if not started:
                    self._record_stage_start(stage_name)
                    started = True
                self._invalidate_downstream(record.sample_id, stage_name, indexes)
                try:
                    result = stage.run(record, self.context)
                except SampleFailure as error:
                    warnings += 1
                    quarantine[record.sample_id] = QuarantineEntry(
                        key=record.sample_id,
                        line_no=record.line_no,
                        sample_id=record.sample_id,
                        audio_path=str(record.item.audio_path),
                        stage=error.stage,
                        category=error.category,
                        message=error.message,
                    )
                    self._purge_sample(record.sample_id, indexes)
                    self._write_indexes(indexes)
                    self._write_quarantine(quarantine_path, quarantine)
                    self._check_limit(quarantine, allowed_bad)
                    continue

                if result.sample_id != record.sample_id:
                    raise ValueError(
                        f"stage {stage_name} returned sample {result.sample_id} for {record.sample_id}"
                    )
                indexes[stage_name][record.sample_id] = {
                    "sample_id": record.sample_id,
                    "signature": signature,
                    "output_paths": [str(path) for path in result.output_paths],
                    "metadata": result.metadata,
                }

            if changed:
                self._write_indexes(indexes)
                index_path = self._index_path(stage_name)
                self.state.complete(stage_name, outputs=[str(index_path)], warning_count=warnings)

        valid_records = [record for record in records if record.sample_id not in quarantine]
        if not valid_records:
            raise QuarantineLimitExceeded("preprocessing left no valid samples")
        self._check_limit(quarantine, allowed_bad)

        required = {"text", "wav32k", "hubert", "sv", "semantic"}
        if selected_stage is None and required.issubset(self.stages):
            write_jsonl(valid_path, [self._valid_row(record) for record in valid_records])

        return PreprocessSummary(
            valid_sample_ids=[record.sample_id for record in valid_records],
            quarantined=list(quarantine.values()),
            allowed_bad=allowed_bad,
            total_records=total_records,
        )

    def _record_stage_start(self, stage_name: str) -> None:
        stage_state = self.state.get(stage_name)
        if stage_state.status is StageStatus.COMPLETED:
            self.state.invalidate(stage_name)
        elif stage_state.status is StageStatus.RUNNING:
            self.state.fail(stage_name, "previous preprocessing run was interrupted")
        self.state.start(stage_name, signature=f"per-sample:{stage_name}")

    def _cache_is_valid(
        self,
        stage: PreprocessStage,
        record: ManifestRecord,
        entry: dict[str, object],
        signature: str,
    ) -> bool:
        if entry.get("signature") != signature:
            return False
        output_paths = entry.get("output_paths")
        if not isinstance(output_paths, list) or not output_paths:
            return False
        paths = [Path(value) for value in output_paths if isinstance(value, str)]
        if len(paths) != len(output_paths):
            return False
        if any(path.name.endswith(".tmp") or not path.is_file() for path in paths):
            return False
        return stage.validate_cached(record, entry, self.context)

    def _invalidate_downstream(
        self,
        sample_id: str,
        stage_name: str,
        indexes: dict[str, dict[str, dict[str, object]]],
    ) -> None:
        for downstream in self.graph.downstream_of(stage_name):
            indexes[downstream].pop(sample_id, None)

    @staticmethod
    def _purge_sample(sample_id: str, indexes: dict[str, dict[str, dict[str, object]]]) -> None:
        for index in indexes.values():
            index.pop(sample_id, None)

    @staticmethod
    def _check_limit(quarantine: dict[str, QuarantineEntry], allowed_bad: int) -> None:
        if len(quarantine) > allowed_bad:
            raise QuarantineLimitExceeded(
                f"bad record limit exceeded: {len(quarantine)} > {allowed_bad}"
            )

    def _index_path(self, stage_name: str) -> Path:
        return self.context.preprocess_dir / stage_name / "index.jsonl"

    def _read_index(self, stage_name: str) -> dict[str, dict[str, object]]:
        path = self._index_path(stage_name)
        if not path.is_file():
            return {}
        entries: dict[str, dict[str, object]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                sample_id = entry["sample_id"]
                if isinstance(sample_id, str):
                    entries[sample_id] = entry
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return entries

    def _write_indexes(self, indexes: dict[str, dict[str, dict[str, object]]]) -> None:
        for stage_name, entries in indexes.items():
            path = self._index_path(stage_name)
            if entries or path.exists():
                write_jsonl(path, [entries[key] for key in sorted(entries)])

    @staticmethod
    def _write_quarantine(path: Path, quarantine: dict[str, QuarantineEntry]) -> None:
        write_jsonl(path, [asdict(entry) for entry in quarantine.values()])

    @staticmethod
    def _valid_row(record: ManifestRecord) -> dict[str, object]:
        return {
            "sample_id": record.sample_id,
            "line_no": record.line_no,
            "audio_path": str(record.item.audio_path),
            "speaker": record.item.speaker,
            "language": record.item.language,
            "text": record.item.text,
        }
