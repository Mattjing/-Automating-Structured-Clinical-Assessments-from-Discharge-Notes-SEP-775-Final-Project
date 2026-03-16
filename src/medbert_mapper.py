"""
Mapper for MedBERT extractor outputs.

This mapper intentionally does not modify the original ``mapper.py``.
It extends base mapping/validation and additionally persists MedBERT evidence
traces into assessment metadata.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.mapper import MDSMapper
from src.mds_schema import MDSAssessment, MDSSchema


class MedBERTMapper(MDSMapper):
    """
    Map MedBERT extraction output to :class:`~mds_schema.MDSAssessment`.

    Expected extra keys in extraction payload:
    * ``"_evidence"``: ``dict[item_id -> list[str]]``
    * ``"_entities"``: ``list[dict]`` from NER stage
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
        """Map extraction and attach MedBERT metadata (evidence/entities)."""
        evidence_map = extraction.get("_evidence", {})
        entities = extraction.get("_entities", [])

        assessment = super().map(
            note_id=note_id,
            subject_id=subject_id,
            hadm_id=hadm_id,
            extraction=extraction,
        )
        assessment.metadata = {
            **assessment.metadata,
            "extractor": "medbert",
            "evidence": evidence_map if isinstance(evidence_map, dict) else {},
            "entities": entities if isinstance(entities, list) else [],
        }
        return assessment
