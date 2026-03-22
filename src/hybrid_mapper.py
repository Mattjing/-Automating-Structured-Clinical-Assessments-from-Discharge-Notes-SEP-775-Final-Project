from __future__ import annotations

from typing import Any, Dict, Optional

from src.mapper import MDSMapper
from src.mds_schema import MDSAssessment, MDSSchema


class HybridMapper(MDSMapper):
    """
    Extends the base mapper by preserving provenance metadata from the
    hybrid extractor.
    """

    def __init__(
        self,
        schema: Optional[MDSSchema] = None,
        strict: bool = False,
    ) -> None:
        super().__init__(schema=schema, strict=strict)

    def map(
        self,
        note_id: str,
        subject_id: str,
        hadm_id: str,
        extraction: Dict[str, Any],
    ) -> MDSAssessment:
        assessment = super().map(
            note_id=note_id,
            subject_id=subject_id,
            hadm_id=hadm_id,
            extraction=extraction,
        )

        assessment.metadata = {
            **getattr(assessment, "metadata", {}),
            "extractor": extraction.get("_extractor", "hybrid"),
            "provenance": extraction.get("_provenance", {}),
            "source_breakdown": extraction.get("_source_breakdown", {}),
        }
        return assessment