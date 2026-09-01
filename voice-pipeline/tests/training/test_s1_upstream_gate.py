from voice_pipeline.training.s1_runtime import S1StepController


def test_s1_upstream_optimizer_gate_matches_official_batch_indices():
    ctl = S1StepController(target_optimizer_steps=2)
    events = [ctl.after_backward(i) for i in range(9)]
    assert events[:4] == [False, False, False, False]
    assert events[4] is True
    assert events[5:8] == [False, False, False]
    assert events[8] is True
    assert ctl.optimizer_steps == 2


def test_s1_target_one_stops_on_first_real_optimizer_step():
    ctl = S1StepController(target_optimizer_steps=1)
    for i in range(4):
        assert ctl.after_backward(i) is False
        assert ctl.should_stop is False
    assert ctl.after_backward(4) is True
    assert ctl.should_stop is True
