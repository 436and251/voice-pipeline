from voice_pipeline.pipeline.graph import StageGraph


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
