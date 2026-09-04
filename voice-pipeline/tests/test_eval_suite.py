from pathlib import Path

import pytest

from voice_pipeline.common.model_bundle import BundleReference
from voice_pipeline.evaluation.config import EvaluationConfig, EvaluationPairing
from voice_pipeline.evaluation.suite import load_evaluation_suite


def _config(tmp_path: Path, suites: dict[str, Path]) -> EvaluationConfig:
    reference = tmp_path / "reference.wav"
    reference.touch()
    return EvaluationConfig(
        run_dir=tmp_path / "runs/speaker",
        project_root=tmp_path,
        model_name="speaker",
        trained_languages=("ja",),
        validated_languages=("zh", "ja", "en"),
        reference=BundleReference(reference, "参照音声です。", "ja"),
        speaker_references=(reference,),
        suites=suites,
        asr_model="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        speaker_model="microsoft/wavlm-base-plus-sv",
        cache_dir=tmp_path / "hf-cache",
        pairing=EvaluationPairing(),
        device="cpu",
        precision="fp32",
    )


def test_suite_uses_file_language_and_stable_ids(tmp_path: Path) -> None:
    suites = {}
    for language in ("zh", "ja", "en", "mixed"):
        path = tmp_path / f"{language}.txt"
        path.write_text(f"{language} sample\n", encoding="utf-8")
        suites[language] = path
    suites["zh"].write_text(
        "\ufeff# comment\n你好，欢迎参加测试。\n\n今天是二〇二六年。\n",
        encoding="utf-8",
    )

    suite = load_evaluation_suite(_config(tmp_path, suites))

    assert [(case.id, case.language, case.text) for case in suite.cases] == [
        ("zh-001", "zh", "你好，欢迎参加测试。"),
        ("zh-002", "zh", "今天是二〇二六年。"),
        ("ja-001", "ja", "ja sample"),
        ("en-001", "en", "en sample"),
        ("mixed-001", "mixed", "mixed sample"),
    ]


def test_suite_rejects_duplicate_sentence_within_language(tmp_path: Path) -> None:
    suites = {}
    for language in ("zh", "ja", "en", "mixed"):
        path = tmp_path / f"{language}.txt"
        path.write_text("same\nsame\n" if language == "ja" else "sample\n", encoding="utf-8")
        suites[language] = path

    with pytest.raises(ValueError, match="duplicate.*ja"):
        load_evaluation_suite(_config(tmp_path, suites))


def test_suite_requires_each_configured_language_to_have_text(tmp_path: Path) -> None:
    suites = {}
    for language in ("zh", "ja", "en", "mixed"):
        path = tmp_path / f"{language}.txt"
        path.write_text("# no case\n" if language == "mixed" else "sample\n", encoding="utf-8")
        suites[language] = path

    with pytest.raises(ValueError, match="mixed.*at least one"):
        load_evaluation_suite(_config(tmp_path, suites))
