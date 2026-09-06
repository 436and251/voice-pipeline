class VoicePipelineError(Exception):
    """Base error for user-facing pipeline failures."""


class ManifestError(VoicePipelineError):
    """Raised when a training manifest violates its contract."""


class EvaluationError(VoicePipelineError):
    """Raised when automatic evaluation cannot produce a valid shortlist."""
