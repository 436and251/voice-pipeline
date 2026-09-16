from pathlib import Path


class ModuleCancelled(RuntimeError):
    pass


class FileCancellationToken:
    def __init__(self, marker: Path) -> None:
        self.marker = Path(marker)

    @property
    def requested(self) -> bool:
        return self.marker.is_file()

    def raise_if_requested(self) -> None:
        if self.requested:
            raise ModuleCancelled("cancellation requested")


__all__ = ["FileCancellationToken", "ModuleCancelled"]
