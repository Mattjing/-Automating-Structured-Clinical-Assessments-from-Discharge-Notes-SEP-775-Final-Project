from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.extractor import LLMExtractor
from src.mds_schema import MDSSchema


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _flatten_structured_payload(payload: Any, prefix: str = "") -> Dict[str, str]:
    """
    Flatten nested structured metadata into a simple key -> string dict.
    Works even if note.metadata contains nested dicts/lists from multiple
    structured sources.
    """
    flat: Dict[str, str] = {}

    if payload is None:
        return flat

    if isinstance(payload, dict):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_structured_payload(value, child_prefix))
        return flat

    if isinstance(payload, list):
        for i, value in enumerate(payload):
            child_prefix = f"{prefix}[{i}]"
            flat.update(_flatten_structured_payload(value, child_prefix))
        return flat

    flat[prefix] = str(payload)
    return flat


class HybridExtractor:
    """
    Hybrid extractor:
    1. extracts from unstructured discharge note with the existing LLMExtractor
    2. extracts additional evidence from structured metadata attached to the note
    3. merges both into one extraction payload

    The output keeps the same shape as the current pipeline expects, with
    optional metadata keys prefixed by "_" so the base mapper can ignore them.
    """

    def __init__(
        self,
        schema: Optional[MDSSchema] = None,
        llm_extractor: Optional[LLMExtractor] = None,
        structured_precedence: bool = True,
    ) -> None:
        self.schema = schema or MDSSchema(section_ids=["I", "N", "O"])
        self.llm_extractor = llm_extractor or LLMExtractor(schema=self.schema, sections=["I", "N", "O"])
        self.structured_precedence = structured_precedence

    def extract(
        self,
        note_text: str,
        note_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        llm_result = self.llm_extractor.extract(note_text, note_metadata=note_metadata)
        structured_result = self._extract_from_structured_metadata(note_metadata or {})

        merged = self._merge_results(llm_result, structured_result)

        merged["_source_breakdown"] = {
            "unstructured": llm_result,
            "structured": structured_result,
        }
        merged["_extractor"] = "hybrid"

        return merged

    def _merge_results(
        self,
        llm_result: Dict[str, Any],
        structured_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        llm_conf = dict(llm_result.get("confidence", {}) or {})
        structured_conf = dict(structured_result.get("confidence", {}) or {})

        item_ids = set(llm_result.keys()) | set(structured_result.keys())
        item_ids.discard("confidence")

        provenance: Dict[str, str] = {}
        merged_conf: Dict[str, float] = {}

        for item_id in item_ids:
            llm_val = llm_result.get(item_id)
            structured_val = structured_result.get(item_id)

            chosen = None
            source = None

            if self.structured_precedence:
                if structured_val is not None:
                    chosen = structured_val
                    source = "structured"
                elif llm_val is not None:
                    chosen = llm_val
                    source = "unstructured"
            else:
                if llm_val is not None:
                    chosen = llm_val
                    source = "unstructured"
                elif structured_val is not None:
                    chosen = structured_val
                    source = "structured"

            if chosen is not None:
                merged[item_id] = chosen
                provenance[item_id] = source or "unknown"
                merged_conf[item_id] = float(
                    structured_conf.get(item_id, 0.99 if source == "structured" else 0.0)
                    if source == "structured"
                    else llm_conf.get(item_id, 0.75)
                )

        merged["confidence"] = merged_conf
        merged["_provenance"] = provenance
        return merged

    def _extract_from_structured_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Best-effort structured extraction from metadata attached by the loader.

        This is intentionally robust to different structured file layouts.
        It scans flattened keys and values and maps them to MDS I/N/O fields
        using clinically relevant keyword heuristics.
        """
        flat = _flatten_structured_payload(metadata)
        result: Dict[str, Any] = {"confidence": {}}

        # Collect all structured text into one searchable corpus
        structured_pairs: List[Tuple[str, str]] = []
        for key, value in flat.items():
            key_n = _norm(key)
            val_n = _norm(value)
            if key_n or val_n:
                structured_pairs.append((key_n, val_n))

        def contains_any(text: str, keywords: Iterable[str]) -> bool:
            return any(k in text for k in keywords)

        # ---------- Section I: diagnoses ----------
        diagnosis_keywords = {
            "I0700": ["hypertension", "htn"],
            "I2000": ["diabetes", "dm", "dm2", "type 2 diabetes", "type ii diabetes", "type 1 diabetes"],
            "I2100": ["septicemia", "sepsis"],
            "I2300": ["urinary tract infection", "uti"],
            "I2500": ["pneumonia"],
            "I2900": ["heart failure", "congestive heart failure", "chf"],
            "I3100": ["atrial fibrillation", "afib", "af"],
            "I4500": ["stroke", "cva", "cerebrovascular accident", "tia"],
            "I4900": ["chronic obstructive pulmonary disease", "copd"],
            "I6200": ["asthma"],
            "I8000": [],  # free-text catch-all
        }

        diagnosis_hits: List[str] = []

        for key, value in structured_pairs:
            row_text = f"{key} {value}"

            is_diagnosis_like = contains_any(
                key,
                ["diag", "diagnosis", "problem", "condition", "dx", "icd", "history", "pmh"],
            )

            if is_diagnosis_like:
                for item_id, keywords in diagnosis_keywords.items():
                    if item_id == "I8000":
                        continue
                    if keywords and contains_any(row_text, keywords) and self.schema.get_item(item_id):
                        result[item_id] = True
                        result["confidence"][item_id] = 0.99
                        diagnosis_hits.append(value)

                # catch-all free text for diagnoses not explicitly mapped
                if value and self.schema.get_item("I8000"):
                    diagnosis_hits.append(value)

        if diagnosis_hits and self.schema.get_item("I8000"):
            uniq = []
            seen = set()
            for x in diagnosis_hits:
                if x and x not in seen:
                    uniq.append(x)
                    seen.add(x)
            result["I8000"] = "; ".join(uniq[:10])
            result["confidence"]["I8000"] = 0.95

        # ---------- Section N: medications ----------
        med_rules = {
            "N0350A": ["insulin"],
            "N0350B": ["anticoagulant", "warfarin", "apixaban", "rivaroxaban", "heparin", "enoxaparin"],
            "N0350C": ["antibiotic", "cef", "cillin", "mycin", "floxacin", "bactrim", "vancomycin"],
            "N0350D": ["diuretic", "furosemide", "lasix", "bumetanide", "spironolactone"],
            "N0350E": ["opioid", "morphine", "oxycodone", "hydromorphone", "tramadol", "fentanyl"],
            "N0415F": ["psychotropic", "antipsychotic", "haloperidol", "quetiapine", "olanzapine"],
        }

        med_day_count_fields = {"N0400A", "N0400B", "N0400C", "N0400D", "N0400E", "N0410A", "N0410B", "N0410C", "N0410D", "N0410E", "N0415F"}

        for key, value in structured_pairs:
            row_text = f"{key} {value}"

            is_med_like = contains_any(
                key,
                ["med", "medication", "drug", "rx", "mar", "prescription"],
            ) or contains_any(
                value,
                ["mg", "tablet", "tab", "capsule", "po", "iv", "bid", "tid", "daily", "qhs"],
            )

            if not is_med_like:
                continue

            for item_id, keywords in med_rules.items():
                if keywords and contains_any(row_text, keywords) and self.schema.get_item(item_id):
                    result[item_id] = True
                    result["confidence"][item_id] = 0.99

            # best-effort day count extraction if the structured source has explicit counts
            for item_id in med_day_count_fields:
                if not self.schema.get_item(item_id):
                    continue

                simple_field_hint = item_id.lower()
                if simple_field_hint in key:
                    match = re.search(r"\b(\d{1,2})\b", value)
                    if match:
                        result[item_id] = int(match.group(1))
                        result["confidence"][item_id] = 0.98

        # ---------- Section O: procedures / treatments ----------
        proc_rules = {
            "O0100C1": ["oxygen"],
            "O0100E1": ["suctioning"],
            "O0100F1": ["tracheostomy"],
            "O0100G1": ["iv", "intravenous", "picc", "central line"],
            "O0100H1": ["dialysis", "hemodialysis", "peritoneal dialysis"],
            "O0100I1": ["transfusion", "blood transfusion"],
            "O0100J1": ["ventilator", "mechanical ventilation", "bipap", "cpap"],
        }

        for key, value in structured_pairs:
            row_text = f"{key} {value}"

            is_proc_like = contains_any(
                key,
                ["procedure", "treatment", "therapy", "device", "support", "resp", "intervention"],
            ) or contains_any(
                value,
                ["oxygen", "iv", "dialysis", "transfusion", "ventilator", "cpap", "bipap"],
            )

            if not is_proc_like:
                continue

            for item_id, keywords in proc_rules.items():
                if keywords and contains_any(row_text, keywords) and self.schema.get_item(item_id):
                    result[item_id] = True
                    result["confidence"][item_id] = 0.99

        return result