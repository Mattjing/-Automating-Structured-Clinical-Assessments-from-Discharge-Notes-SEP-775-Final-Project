"""
Mapper module — converts raw LLM extraction results to :class:`~mds_schema.MDSAssessment` objects.

The mapper validates extracted values against the MDS schema, logs any
discrepancies, and packages the results into an :class:`~mds_schema.MDSAssessment`
that can be serialised to JSON, CSV, or Excel.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.mds_schema import MDSAssessment, MDSItemType, MDSSchema

logger = logging.getLogger(__name__)


class MDSMapper:
    """
    Maps raw LLM extraction results to :class:`~mds_schema.MDSAssessment` objects.

    The mapper:
    1. Validates each extracted value against the schema (e.g. checks that
       select-field codes are valid).
    2. Discards invalid values and logs a warning.
    3. Stores the remaining values — along with per-item confidence scores —
       in an :class:`~mds_schema.MDSAssessment`.

    Parameters
    ----------
    schema : MDSSchema, optional
        The MDS schema to validate against.  A new default schema is created
        if not provided.
    strict : bool
        When ``True``, raise :exc:`ValueError` for invalid values instead of
        logging warnings and discarding them.
    """

    def __init__(
        self,
        schema: Optional[MDSSchema] = None,
        strict: bool = False,
    ) -> None:
        self.schema = schema or MDSSchema()
        self.strict = strict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map(
        self,
        note_id: str,
        subject_id: str,
        hadm_id: str,
        extraction: Dict[str, Any],
    ) -> MDSAssessment:
        """
        Map a raw extraction dict to an :class:`~mds_schema.MDSAssessment`.

        Parameters
        ----------
        note_id : str
            Identifier of the source discharge note.
        subject_id : str
            Patient identifier.
        hadm_id : str
            Hospital admission identifier.
        extraction : dict
            Raw extraction dict as returned by :meth:`~extractor.LLMExtractor.extract`.
            May contain a ``"confidence"`` key with per-item scores.

        Returns
        -------
        MDSAssessment
        """
        confidence_map: Dict[str, float] = extraction.get("confidence", {})
        assessment = MDSAssessment(
            note_id=note_id,
            subject_id=subject_id,
            hadm_id=hadm_id,
        )

        for item_id, value in extraction.items():
            if item_id == "confidence" or value is None:
                continue

            item = self.schema.get_item(item_id)
            if item is None:
                logger.debug("Unknown MDS item id %r — skipping.", item_id)
                continue

            validated = self._validate(item_id, value, item.item_type, item.option_codes())
            if validated is None:
                continue

            conf = confidence_map.get(item_id, 1.0)
            assessment.set_field(item_id, validated, conf)

        return assessment

    def map_batch(
        self,
        records: List[Dict[str, Any]],
        extractions: List[Dict[str, Any]],
    ) -> List[MDSAssessment]:
        """
        Map a list of extractions to a list of assessments.

        Parameters
        ----------
        records : list of dict
            Each dict must have keys ``note_id``, ``subject_id``, ``hadm_id``.
        extractions : list of dict
            Corresponding raw extraction dicts.

        Returns
        -------
        list of MDSAssessment
        """
        if len(records) != len(extractions):
            raise ValueError(
                f"records ({len(records)}) and extractions ({len(extractions)}) "
                "must have the same length."
            )
        assessments: List[MDSAssessment] = []
        for record, extraction in zip(records, extractions):
            assessment = self.map(
                note_id=record["note_id"],
                subject_id=record["subject_id"],
                hadm_id=record["hadm_id"],
                extraction=dict(extraction),  # avoid mutating the original
            )
            assessments.append(assessment)
        return assessments

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(
        self,
        item_id: str,
        value: Any,
        item_type: MDSItemType,
        option_codes: List[str],
    ) -> Optional[Any]:
        """
        Validate *value* for an item of *item_type*.

        Returns the (possibly coerced) value on success, or ``None`` if
        invalid (when ``self.strict=False``) / raises ``ValueError`` (when
        ``self.strict=True``).
        """
        try:
            if item_type == MDSItemType.BOOLEAN:
                return self._validate_boolean(item_id, value)
            if item_type == MDSItemType.INTEGER:
                return self._validate_integer(item_id, value)
            if item_type == MDSItemType.DATE:
                return str(value)
            if item_type == MDSItemType.SELECT:
                return self._validate_select(item_id, value, option_codes)
            if item_type == MDSItemType.MULTI:
                return self._validate_multi(item_id, value, option_codes)
            # TEXT
            return str(value)
        except ValueError as exc:
            if self.strict:
                raise
            logger.warning("Validation failed for %r: %s — value discarded.", item_id, exc)
            return None

    @staticmethod
    def _validate_boolean(item_id: str, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in {"true", "yes", "1"}:
                return True
            if lower in {"false", "no", "0"}:
                return False
        raise ValueError(
            f"Expected boolean for {item_id!r}, got {value!r}"
        )

    @staticmethod
    def _validate_integer(item_id: str, value: Any) -> int:
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            raise ValueError(f"Expected integer for {item_id!r}, got {value!r}")

    @staticmethod
    def _validate_select(
        item_id: str, value: Any, option_codes: List[str]
    ) -> str:
        code = str(value).strip()
        if option_codes and code not in option_codes:
            raise ValueError(
                f"Invalid code {code!r} for select item {item_id!r}. "
                f"Valid codes: {option_codes}"
            )
        return code

    @staticmethod
    def _validate_multi(
        item_id: str, value: Any, option_codes: List[str]
    ) -> List[str]:
        if isinstance(value, list):
            codes = [str(v).strip() for v in value]
        elif isinstance(value, str):
            codes = [value.strip()]
        else:
            raise ValueError(
                f"Expected list or string for multi-select item {item_id!r}, got {type(value).__name__}"
            )
        if option_codes:
            invalid = [c for c in codes if c not in option_codes]
            if invalid:
                raise ValueError(
                    f"Invalid codes {invalid} for multi-select item {item_id!r}. "
                    f"Valid codes: {option_codes}"
                )
        return codes
