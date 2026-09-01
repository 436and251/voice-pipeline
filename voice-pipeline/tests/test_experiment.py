from pathlib import Path

from voice_pipeline.training.experiment import Experiment


def test_experiment_creates_isolated_layout(tmp_path: Path):
    exp = Experiment.create("speaker_001", tmp_path)
    assert exp.root == tmp_path / "speaker_001"
    assert exp.input_dir.is_dir()
    assert exp.preprocess_dir.is_dir()
    assert exp.s1_dir.is_dir()
    assert exp.s2_dir.is_dir()
    assert exp.evaluation_dir.is_dir()
    assert exp.export_dir.is_dir()
