from pathlib import Path

import pytest

from voice_pipeline.module_api.cancellation import FileCancellationToken, ModuleCancelled


def test_file_cancellation_token_tracks_marker(tmp_path: Path) -> None:
    marker = tmp_path / "cancel.requested"
    token = FileCancellationToken(marker)

    assert token.requested is False
    token.raise_if_requested()

    marker.touch()
    assert token.requested is True
    with pytest.raises(ModuleCancelled, match="cancellation requested"):
        token.raise_if_requested()
