from pathlib import Path

from voice_pipeline.evaluation.base import MetricResult
from voice_pipeline.evaluation.config import EvaluationConstraints, EvaluationRanking
from voice_pipeline.evaluation.ranking import assign_anonymous_ids, rank_candidates
from voice_pipeline.evaluation.report import write_report, write_results
from voice_pipeline.evaluation.runner import CandidateEvaluation, EvaluatedSample


def _evaluation(
    key: str,
    *,
    similarity: dict[str, float],
    pronunciation: dict[str, float] | None = None,
    language_score: float = 1.0,
    prosody_score: float = 1.0,
    status: str = "completed",
) -> CandidateEvaluation:
    pronunciation = pronunciation or {"zh": 0.05, "ja": 0.05, "en": 0.05}
    by_language = {
        language: {
            "speaker_similarity": value,
            "pronunciation": pronunciation[language],
            "language_consistency": language_score,
            "prosody": prosody_score,
        }
        for language, value in similarity.items()
    }
    samples = tuple(
        EvaluatedSample(
            f"{language}-001",
            language,
            "completed",
            None,
            {
                "prosody": MetricResult(
                    "prosody",
                    prosody_score,
                    {"all_silent": False, "severe_clipping": False},
                    True,
                )
            },
        )
        for language in similarity
    )
    return CandidateEvaluation(key, status, samples, by_language, {"worst_similarity": min(similarity.values())})


def _constraints() -> EvaluationConstraints:
    return EvaluationConstraints(
        min_speaker_similarity={"zh": 0.8, "ja": 0.8, "en": 0.8},
        max_cer={"zh": 0.2, "ja": 0.2},
        max_wer={"en": 0.25},
        min_language_consistency={"zh": 0.5, "ja": 0.5, "en": 0.5, "mixed": 0.5},
    )


def test_high_average_cannot_hide_failed_english_similarity() -> None:
    bad_english = _evaluation("pair-a", similarity={"zh": 0.95, "ja": 0.96, "en": 0.60})
    balanced = _evaluation("pair-b", similarity={"zh": 0.84, "ja": 0.86, "en": 0.83})

    ranked = rank_candidates((bad_english, balanced), _constraints(), EvaluationRanking())

    assert ranked[0].pair_key == "pair-b"
    assert ranked[0].eligible is True
    assert ranked[1].eligible is False
    assert ranked[1].failures[0].metric == "speaker_similarity.en"


def test_hard_constraints_reject_bad_pronunciation_and_silent_audio() -> None:
    result = _evaluation(
        "bad",
        similarity={"zh": 0.9, "ja": 0.9, "en": 0.9},
        pronunciation={"zh": 0.4, "ja": 0.05, "en": 0.05},
    )
    silent = MetricResult("prosody", 0.0, {"all_silent": True, "severe_clipping": False}, True)
    result.samples[0].metrics["prosody"] = silent

    ranked = rank_candidates((result,), _constraints(), EvaluationRanking())

    assert ranked[0].eligible is False
    assert {failure.metric for failure in ranked[0].failures} == {"pronunciation.zh", "audio.all_silent"}


def test_speaker_priority_wins_after_both_candidates_pass() -> None:
    stronger_voice = _evaluation(
        "voice",
        similarity={"zh": 0.91, "ja": 0.92, "en": 0.90},
        pronunciation={"zh": 0.19, "ja": 0.19, "en": 0.24},
    )
    clearer_words = _evaluation(
        "words",
        similarity={"zh": 0.82, "ja": 0.83, "en": 0.81},
        pronunciation={"zh": 0.0, "ja": 0.0, "en": 0.0},
    )

    ranked = rank_candidates((clearer_words, stronger_voice), _constraints(), EvaluationRanking())

    assert [candidate.pair_key for candidate in ranked] == ["voice", "words"]
    assert ranked[0].components["speaker"] > ranked[1].components["speaker"]


def test_anonymous_ids_are_stable_and_only_assigned_to_eligible_candidates() -> None:
    results = (
        _evaluation("one", similarity={"zh": 0.9, "ja": 0.9, "en": 0.9}),
        _evaluation("two", similarity={"zh": 0.88, "ja": 0.88, "en": 0.88}),
        _evaluation("bad", similarity={"zh": 0.9, "ja": 0.9, "en": 0.2}),
    )
    ranked = rank_candidates(results, _constraints(), EvaluationRanking())

    anonymous = assign_anonymous_ids(ranked, limit=3)

    assert [(item.pair_key, item.public_id) for item in anonymous] == [
        ("one", "candidate_A"),
        ("two", "candidate_B"),
    ]


def test_reports_preserve_raw_metrics_and_failure_reasons(tmp_path: Path) -> None:
    ranked = rank_candidates(
        (_evaluation("bad", similarity={"zh": 0.9, "ja": 0.9, "en": 0.2}),),
        _constraints(),
        EvaluationRanking(),
    )

    json_path = write_results(tmp_path / "results.json", ranked)
    markdown_path = write_report(tmp_path / "report.md", ranked)

    assert '"speaker_similarity": 0.2' in json_path.read_text(encoding="utf-8")
    report = markdown_path.read_text(encoding="utf-8")
    assert "pair-a" not in report
    assert "speaker_similarity.en" in report
    assert "pair-a" not in report
    assert "bad" in report
