from voice_pipeline.pipeline.graph import StageGraph
import pytest


def test_invalidate_downstream_only():
    graph = StageGraph({
        "text": set(),
        "wav32k": set(),
        "hubert": {"wav32k"},
        "sv": {"wav32k"},
        "semantic": {"hubert"},
        "s1": {"text", "semantic"},
        "s2": {"text", "wav32k", "hubert", "sv"},
        "evaluate": {"s1", "s2"},
        "export": {"evaluate"},
    })

    assert graph.downstream_of("wav32k") == {
        "hubert", "sv", "semantic", "s1", "s2", "evaluate", "export"
    }
    assert "text" not in graph.downstream_of("wav32k")


def test_topological_order_is_deterministic_and_target_includes_dependencies():
    graph = StageGraph({
        "text": set(),
        "wav32k": set(),
        "hubert": {"wav32k"},
        "semantic": {"hubert"},
    })

    assert graph.topological_order() == ["text", "wav32k", "hubert", "semantic"]
    assert graph.topological_order("semantic") == ["wav32k", "hubert", "semantic"]


def test_unknown_dependency_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        StageGraph({"semantic": {"missing"}})


def test_cycle_is_rejected_with_stage_names():
    with pytest.raises(ValueError, match=r"hubert.*semantic|semantic.*hubert"):
        StageGraph({"hubert": {"semantic"}, "semantic": {"hubert"}})
