"""
Mapper module — converts raw LLM extraction results to :class:`~mds_schema.MDSAssessment` objects.

The mapper validates extracted values against the MDS schema, logs any
discrepancies, and packages the results into an :class:`~mds_schema.MDSAssessment`
that can be serialised to JSON, CSV, or Excel.

Improvements over v1
---------------------
* Per-item evidence strings extracted and stored in ``assessment.metadata["evidence"]``
* Per-item status tracking (``"valid"`` / ``"invalid"`` / ``"missing"``)
  stored in ``assessment.metadata["item_status"]``
* Section-level summary statistics logged after each mapping
* ``to_json()`` and ``to_csv()`` convenience methods on the returned assessment
"""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any, Dict, List, Optional

from src.mds_schema import MDSAssessment, MDSItem, MDSItemType, MDSSchema

logger = logging.getLogger(__name__)


class MDSMapper:
    """
    Maps raw LLM extraction results to :class:`~mds_schema.MDSAssessment` objects.

    The mapper:
    1. Extracts per-item evidence strings (if present in the extraction dict).
    2. Validates each extracted value against the schema (e.g. checks that
       select-field codes are valid).
    3. Records item status: ``"valid"`` (passed validation), ``"invalid"``
       (failed validation and discarded), or ``"missing"`` (not in extraction).
    4. Stores the remaining values — along with per-item confidence scores —
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
            May contain a ``"confidence"`` key with per-item scores and optionally
            per-item ``"evidence"`` strings.

        Returns
        -------
        MDSAssessment
            The assessment object with validated fields, confidence scores,
            and metadata including evidence and item status.
        """
        confidence_map: Dict[str, float] = extraction.get("confidence", {})
        evidence_map: Dict[str, str] = extraction.get("evidence", {})
        assessment = MDSAssessment(
            note_id=note_id,
            subject_id=subject_id,
            hadm_id=hadm_id,
        )

        # Track per-item status and section-level counts
        item_status: Dict[str, str] = {}
        section_stats: Dict[str, Dict[str, int]] = {}
        items_in_scope = self._get_all_schema_items()

        for item in items_in_scope:
            item_id = item.item_id
            section_id = self._item_section(item_id)

            # Initialize section stats
            if section_id not in section_stats:
                section_stats[section_id] = {
                    "total": 0, "valid": 0, "invalid": 0, "missing": 0,
                }
            section_stats[section_id]["total"] += 1

            # Check if item was extracted
            if item_id not in extraction or extraction[item_id] is None:
                item_status[item_id] = "missing"
                section_stats[section_id]["missing"] += 1
                continue

            value = extraction[item_id]
            validated = self._validate(
                item_id, value, item.item_type, item.option_codes(),
            )

            if validated is None:
                item_status[item_id] = "invalid"
                section_stats[section_id]["invalid"] += 1
                continue

            item_status[item_id] = "valid"
            section_stats[section_id]["valid"] += 1

            conf = confidence_map.get(item_id, 1.0)
            assessment.set_field(item_id, validated, conf)

        # Store metadata
        assessment.metadata["item_status"] = item_status
        assessment.metadata["section_stats"] = section_stats
        if evidence_map:
            assessment.metadata["evidence"] = {
                k: v for k, v in evidence_map.items()
                if k in item_status and item_status[k] == "valid"
            }

        # Log summary
        self._log_summary(note_id, section_stats)

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
    # Serialization helpers (operate on MDSAssessment)
    # ------------------------------------------------------------------

    @staticmethod
    def to_json(assessment: MDSAssessment, indent: int = 2) -> str:
        """Serialize an assessment to a JSON string."""
        return json.dumps(assessment.to_dict(), indent=indent, default=str)

    @staticmethod
    def to_csv_row(assessment: MDSAssessment) -> str:
        """Serialize an assessment to a single CSV row string."""
        flat = assessment.to_flat_dict()
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=sorted(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
        return output.getvalue()

    @staticmethod
    def assessments_to_csv(assessments: List[MDSAssessment]) -> str:
        """Serialize a list of assessments to a CSV string."""
        if not assessments:
            return ""
        rows = [a.to_flat_dict() for a in assessments]
        all_keys: List[str] = []
        seen: set = set()
        # Stable column ordering: identifiers first, then item IDs sorted
        for key in ["note_id", "subject_id", "hadm_id"]:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)
        for row in rows:
            for key in sorted(row.keys()):
                if key not in seen:
                    all_keys.append(key)
                    seen.add(key)

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return output.getvalue()

    @staticmethod
    def assessments_to_dataframe(assessments: List[MDSAssessment]) -> "pd.DataFrame":
        """Convert assessments to a pandas DataFrame."""
        import pandas as pd
        rows = [a.to_flat_dict() for a in assessments]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_all_schema_items(self) -> List[MDSItem]:
        """Return all items in the schema."""
        return self.schema.all_items()

    @staticmethod
    def _item_section(item_id: str) -> str:
        """Infer section letter from item ID (e.g. 'I0400' → 'I')."""
        for i, ch in enumerate(item_id):
            if ch.isdigit():
                return item_id[:i].upper()
        return item_id[0].upper()

    def _log_summary(
        self,
        note_id: str,
        section_stats: Dict[str, Dict[str, int]],
    ) -> None:
        """Log a concise per-section mapping summary."""
        parts: List[str] = []
        total_valid = 0
        total_invalid = 0
        total_missing = 0

        for sec_id in sorted(section_stats.keys()):
            s = section_stats[sec_id]
            total_valid += s["valid"]
            total_invalid += s["invalid"]
            total_missing += s["missing"]
            if s["valid"] > 0 or s["invalid"] > 0:
                parts.append(
                    f"{sec_id}:{s['valid']}ok/{s['invalid']}bad/{s['missing']}miss"
                )

        summary = " | ".join(parts) if parts else "no items mapped"
        logger.info(
            "MDSMapper note=%s: %d valid, %d invalid, %d missing — %s",
            note_id, total_valid, total_invalid, total_missing, summary,
        )

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
                return self._validate_date(item_id, value)
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
            if lower in {"true", "yes", "1", "present", "confirmed"}:
                return True
            if lower in {"false", "no", "0", "absent", "none"}:
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
    def _validate_date(item_id: str, value: Any) -> str:
        """Validate and normalize a date value."""
        s = str(value).strip()
        if not s or s.lower() in {"none", "null", "n/a", "---"}:
            raise ValueError(f"Empty or null date for {item_id!r}")
        return s

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
            # Handle comma-separated strings
            if "," in value:
                codes = [v.strip() for v in value.split(",") if v.strip()]
            else:
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
