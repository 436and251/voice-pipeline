from __future__ import annotations

import sys


def main() -> None:
    """Keep module discovery light while preserving the existing CLI."""
    if sys.argv[1:3] == ["module", "describe"]:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        from voice_pipeline.cli.module import app

        app(args=sys.argv[2:], prog_name="voice-pipeline module")
        return

    from voice_pipeline.cli.main import app

    app()


__all__ = ["main"]
