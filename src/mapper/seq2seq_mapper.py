"""
Mapper for Seq2SeqExtractor output — Option 1.

Extends :class:`~mapper.mapper.MDSMapper` with:

* Stripping of ``_raw_output`` and ``_model_name`` before base validation
* Storage of the raw generated text and model name in assessment metadata
* Logging of the item count mapped per note

The extraction format produced by :class:`~extractor.seq2seq_extractor.Seq2SeqExtractor`
is flat ``{item_id: value, "confidence": {item_id: float}}`` — identical to the
format expected by :class:`~mapper.mapper.MDSMapper`, so all schema validation
is inherited unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.mapper.mapper import MDSMapper
from src.mds_schema import MDSAssessment, MDSSchema

logger = logging.getLogger(__name__)


class Seq2SeqMapper(MDSMapper):
    """
    Maps :class:`~extractor.seq2seq_extractor.Seq2SeqExtractor` output to
    :class:`~mds_schema.MDSAssessment`.

    Inherits all schema validation from :class:`~mapper.mapper.MDSMapper`.
    Additionally strips seq2seq reserved keys from the extraction dict before
    validation and stores them in ``assessment.metadata``.

    Extra metadata keys set on the returned assessment
    --------------------------------------------------
    ``seq2seq_raw_output``
        First 1,000 characters of the raw decoder output.
    ``seq2seq_model``
        Hugging Face model ID or fine-tuned checkpoint path.
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
        """
        Map seq2seq extraction output to an :class:`~mds_schema.MDSAssessment`.

        Reserved keys ``"_raw_output"`` and ``"_model_name"`` are extracted
        before schema validation so they do not interfere with item mapping.
        """
        # Pop reserved metadata keys before passing to base mapper
        raw_output = extraction.pop("_raw_output", "")
        model_name = extraction.pop("_model_name", "")

        assessment = super().map(
            note_id=note_id,
            subject_id=subject_id,
            hadm_id=hadm_id,
            extraction=extraction,
        )

        # Attach seq2seq-specific metadata (merge with base metadata)
        assessment.metadata.update({
            "extractor":          "seq2seq",
            "seq2seq_model":      model_name,
            "seq2seq_raw_output": raw_output[:1000],
        })

        logger.debug(
            "Seq2SeqMapper: mapped %d fields for note %s (model=%s).",
            len(assessment.fields),
            note_id,
            model_name,
        )
        return assessment
