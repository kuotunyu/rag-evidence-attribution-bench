"""Offline, human-only annotation tooling.

The package validates, projects, schedules, and analyzes human decisions. It never infers
or fills an annotation from challenge transformation metadata.
"""

from rag_evidence.annotation.models import (
    AdjudicationRecord,
    AnnotationAmendment,
    AnswerabilityAnnotation,
    BlindTask,
    CitationAnnotation,
    EligibilityArtifact,
    EligibilityRecord,
)

__all__ = [
    "AdjudicationRecord",
    "AnnotationAmendment",
    "AnswerabilityAnnotation",
    "BlindTask",
    "CitationAnnotation",
    "EligibilityArtifact",
    "EligibilityRecord",
]
