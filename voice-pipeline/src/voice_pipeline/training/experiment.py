from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Experiment:
    root: Path
    input_dir: Path
    preprocess_dir: Path
    s1_dir: Path
    s2_dir: Path
    evaluation_dir: Path
    export_dir: Path

    @classmethod
    def create(cls, name: str, output_root: Path) -> "Experiment":
        root = output_root / name
        paths = {
            "input_dir": root / "input",
            "preprocess_dir": root / "preprocess",
            "s1_dir": root / "training" / "s1",
            "s2_dir": root / "training" / "s2",
            "evaluation_dir": root / "evaluation",
            "export_dir": root / "export",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return cls(root=root, **paths)
