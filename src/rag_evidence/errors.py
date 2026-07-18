"""Exception hierarchy. Every expected failure raises a RagEvidenceError subclass;
the CLI maps those to exit code 1 with a clean message, anything else is a bug."""

from __future__ import annotations


class RagEvidenceError(Exception):
    """Base class for all expected errors in this package."""


class ConfigError(RagEvidenceError):
    """Invalid or inconsistent configuration."""


class DataError(RagEvidenceError):
    """Dataset loading or normalization failed."""


class FingerprintMismatchError(RagEvidenceError):
    """Dataset content does not match the committed split manifest."""

    def __init__(self, question_id: str, expected: str, actual: str) -> None:
        self.question_id = question_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"fingerprint mismatch for question {question_id}: "
            f"expected {expected[:12]}…, got {actual[:12]}… — the downloaded dataset "
            "content differs from the committed manifest; do NOT proceed."
        )


class ArtifactError(RagEvidenceError):
    """Reading or writing a results artifact failed."""


class ResumeConflictError(RagEvidenceError):
    """--resume attempted with a different scientific config than the original run."""


class UpstreamMissingError(RagEvidenceError):
    """A stage requires the output of an earlier stage that has not been run."""


class GpuRequiredError(RagEvidenceError):
    """A CUDA device is required for this stage but none is available."""


class InvalidOutputError(RagEvidenceError):
    """A method produced structurally invalid output (NaN scores, wrong ID set, …)."""


class SampleNotFoundError(RagEvidenceError):
    """Requested sample_id is not present in the loaded results."""
