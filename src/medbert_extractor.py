"""
MedBERT-based extractor for MDS 3.0 fields.

This module keeps the original GPT extractor untouched and adds a more
"scientific" alternative based on biomedical NER + deterministic mapping
rules:

1) run a MedBERT/clinical-BERT token-classification model,
2) collect diagnosis/medication/procedure entities,
3) map evidence to MDS 3.0 Section I/N/O item IDs,
4) return structured extraction output with confidence and evidence traces.

Notes
-----
* Use a token-classification checkpoint for best results (e.g. biomedical NER
  models on Hugging Face).
* The default model is configurable via ``model_name``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

from src.mds_schema import MDSItem, MDSItemType, MDSSchema
from src.preprocessor import build_extraction_context

logger = logging.getLogger(__name__)

_DEFAULT_AUTO_PATTERN_PATH = Path(__file__).resolve().parents[1] / "config" / "medbert_patterns.auto.json"


_GENERIC_SCHEMA_TOKENS: set[str] = {
    "active",
    "additional",
    "calendar",
    "category",
    "condition",
    "conditions",
    "code",
    "days",
    "date",
    "disease",
    "disorder",
    "disorders",
    "distinct",
    "drug",
    "icd",
    "last",
    "medical",
    "medication",
    "medications",
    "minutes",
    "none",
    "number",
    "other",
    "part",
    "primary",
    "programs",
    "reason",
    "received",
    "review",
    "risk",
    "services",
    "therapy",
    "total",
    "vaccine",
    "within",
}

_N0415_NONE_PATTERNS: Tuple[str, ...] = (
    r"\bnone of the above\b",
    r"\bno high[-\s]?risk drug classes?\b",
)

_N0450A_PATTERNS: Tuple[str, ...] = (
    r"\bantipsychotic(?: medications?)? (?:received|used|given)\b",
    r"\breceived antipsychotic\b",
)
_N0450B_PATTERNS: Tuple[str, ...] = (
    r"\bgradual dose reduction\b",
    r"\bGDR\b",
    r"\bdose reduction attempted\b",
)
_N0450D_PATTERNS: Tuple[str, ...] = (
    r"\bclinically contraindicated\b",
    r"\bphysician documented\b",
)

_INSULIN_PATTERNS: Tuple[str, ...] = (r"\binsulin\b", r"\bsliding scale\b")
_INJECTION_PATTERNS: Tuple[str, ...] = (r"\binjection\w*\b", r"\binjected\b", r"\bsubcutaneous\b")
_INSULIN_DOSE_CHANGE_PATTERNS: Tuple[str, ...] = (
    r"\binsulin dose change\w*\b",
    r"\binsulin dose adjusted\b",
    r"\binsulin titrat\w*\b",
    r"\bsliding scale adjusted\b",
)
_DRUG_REVIEW_PATTERNS: Tuple[str, ...] = (
    r"\bdrug regimen review\b",
    r"\bmedication reconciliation\b",
    r"\bmedications reviewed\b",
)
_MEDICATION_FOLLOWUP_PATTERNS: Tuple[str, ...] = (
    r"\bphysician(?:[-\s]?designee)? contacted\b",
    r"\bmedication follow[-\s]?up\b",
    r"\brecommend(?:ed|ation) actions? completed\b",
)
_MEDICATION_INTERVENTION_PATTERNS: Tuple[str, ...] = (
    r"\bmedication intervention\b",
    r"\bclinically significant medication issues?\b",
    r"\bactions? completed by midnight of the next calendar day\b",
)


class MedBERTExtractor:
    """
    Biomedical NER + rules extractor for MDS Section I/N/O.

    Parameters
    ----------
    schema : MDSSchema, optional
        MDS schema. Defaults to ``MDSSchema()``.
    model_name : str
        Hugging Face model id for token-classification.
    sections : list of str, optional
        Section filter (defaults to ``["I", "N", "O"]``).
    preprocess_input : bool
        Whether to run focused preprocessing before NER.
    ner_pipeline : Any, optional
        Injected transformers pipeline (useful for testing/mocking).
    """

    def __init__(
        self,
        schema: Optional[MDSSchema] = None,
        model_name: str = "d4data/biomedical-ner-all",
        sections: Optional[List[str]] = None,
        preprocess_input: bool = True,
        ner_pipeline: Optional[Any] = None,
        auto_pattern_path: Optional[str] = None,
    ) -> None:
        self.schema = schema or MDSSchema()
        self.model_name = model_name
        self.sections = [section.upper() for section in sections] if sections else ["I", "N", "O"]
        self.preprocess_input = preprocess_input

        # Build baseline rules from schema labels with item-type scoping.
        self.diagnosis_patterns: Dict[str, Tuple[str, ...]] = _schema_label_patterns(
            self.schema,
            "I",
            item_filter=lambda item: item.item_type == MDSItemType.BOOLEAN,
        )
        self.medication_patterns: Dict[str, Tuple[str, ...]] = _schema_label_patterns(
            self.schema,
            "N",
            item_filter=lambda item: item.item_id.startswith("N0415") and item.item_id != "N0415Z",
        )
        self.treatment_patterns: Dict[str, Tuple[str, ...]] = _schema_label_patterns(
            self.schema,
            "O",
            item_filter=lambda item: item.item_type == MDSItemType.BOOLEAN,
        )
        self._load_auto_patterns(auto_pattern_path)

        self._ner_pipeline = ner_pipeline or self._build_ner_pipeline()

    def extract(self, note_text: str) -> Dict[str, Any]:
        """Extract MDS fields from a single discharge note."""
        prepared = self._prepare_note_text(note_text)
        entities = self._extract_entities(prepared)

        result: Dict[str, Any] = {}
        confidence: Dict[str, float] = {}
        evidence: Dict[str, List[str]] = {}

        items = self._get_items_to_extract()
        item_ids = {item.item_id for item in items}

        for item_id, patterns in self.diagnosis_patterns.items():
            if item_id not in item_ids:
                continue
            hit, snippets = _find_pattern_hits(prepared, patterns)
            if hit:
                result[item_id] = True
                evidence[item_id] = snippets
                confidence[item_id] = _confidence_from_entities(entities, snippets, base=0.78)

        result.update(self._extract_medication_items(prepared, entities, item_ids, confidence, evidence))
        result.update(self._extract_treatment_items(prepared, entities, item_ids, confidence, evidence))
        self._fill_i8000_from_entities(result, entities, item_ids, confidence)

        result["confidence"] = confidence
        result["_evidence"] = evidence
        result["_entities"] = entities
        return result

    def extract_batch(self, notes: List[str]) -> List[Dict[str, Any]]:
        """Extract MDS fields for multiple notes."""
        return [self.extract(note) for note in notes]

    def _build_ner_pipeline(self) -> Any:
        """Build a Hugging Face NER pipeline."""
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for MedBERT extraction. "
                "Install it with: pip install transformers torch"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        return pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )

    def _prepare_note_text(self, note_text: str) -> str:
        if not self.preprocess_input:
            return note_text or ""
        return build_extraction_context(note_text, sections=self.sections)

    def _get_items_to_extract(self) -> List[MDSItem]:
        items: List[MDSItem] = []
        for section_id in self.sections:
            section = self.schema.get_section(section_id)
            if section:
                items.extend(section.items)
        return items

    def _extract_entities(self, note_text: str) -> List[Dict[str, Any]]:
        if not note_text.strip():
            return []

        raw_entities = cast(List[Dict[str, Any]], self._ner_pipeline(note_text[:12000]))
        entities: List[Dict[str, Any]] = []
        for ent in raw_entities:
            text = str(ent.get("word", "")).strip()
            if not text:
                continue
            entities.append(
                {
                    "text": text,
                    "label": str(ent.get("entity_group") or ent.get("entity") or "").upper(),
                    "score": float(ent.get("score", 0.0)),
                    "start": int(ent.get("start", -1)),
                    "end": int(ent.get("end", -1)),
                }
            )
        return entities

    def _extract_medication_items(
        self,
        note_text: str,
        entities: List[Dict[str, Any]],
        item_ids: set[str],
        confidence: Dict[str, float],
        evidence: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        for item_id, patterns in self.medication_patterns.items():
            if item_id not in item_ids:
                continue
            hit, snippets = _find_pattern_hits(note_text, patterns)
            if hit:
                # N0415x items are MULTI ("1" = Is taking); legacy integer day count removed
                result[item_id] = ["1"]
                evidence[item_id] = snippets
                confidence[item_id] = _confidence_from_entities(entities, snippets, base=0.72)

        if "N0415Z" in item_ids:
            hit, snippets = _find_pattern_hits(note_text, _N0415_NONE_PATTERNS)
            if hit:
                result["N0415Z"] = True
                evidence["N0415Z"] = snippets
                confidence["N0415Z"] = _confidence_from_entities(entities, snippets, base=0.7)

        if "N0350A" in item_ids:
            hit, snippets = _find_pattern_hits(note_text, _INSULIN_PATTERNS)
            if hit:
                result["N0350A"] = _infer_days_from_snippets(snippets) or 7
                evidence["N0350A"] = snippets
                confidence["N0350A"] = _confidence_from_entities(entities, snippets, base=0.74)

        if "N0300" in item_ids:
            hit, snippets = _find_pattern_hits(note_text, _INJECTION_PATTERNS)
            if hit:
                result["N0300"] = _infer_days_from_snippets(snippets) or 1
                evidence["N0300"] = snippets
                confidence["N0300"] = _confidence_from_entities(entities, snippets, base=0.68)

        if "N0350B" in item_ids:
            hit, snippets = _find_pattern_hits(note_text, _INSULIN_DOSE_CHANGE_PATTERNS)
            if hit:
                result["N0350B"] = True
                evidence["N0350B"] = snippets
                confidence["N0350B"] = _confidence_from_entities(entities, snippets, base=0.8)

        if "N0450A" in item_ids:
            hit, snippets = _find_pattern_hits(note_text, _N0450A_PATTERNS)
            if hit:
                result["N0450A"] = "1"
                evidence["N0450A"] = snippets
                confidence["N0450A"] = _confidence_from_entities(entities, snippets, base=0.76)

        if "N0450B" in item_ids:
            hit, snippets = _find_pattern_hits(note_text, _N0450B_PATTERNS)
            if hit:
                result["N0450B"] = True
                evidence["N0450B"] = snippets
                confidence["N0450B"] = _confidence_from_entities(entities, snippets, base=0.76)

        if "N0450D" in item_ids:
            hit, snippets = _find_pattern_hits(note_text, _N0450D_PATTERNS)
            if hit:
                result["N0450D"] = True
                evidence["N0450D"] = snippets
                confidence["N0450D"] = _confidence_from_entities(entities, snippets, base=0.76)

        if "N2001" in item_ids:
            hit, snippets = _find_pattern_hits(note_text, _DRUG_REVIEW_PATTERNS)
            if hit:
                result["N2001"] = True
                evidence["N2001"] = snippets
                confidence["N2001"] = _confidence_from_entities(entities, snippets, base=0.82)

        if "N2003" in item_ids:
            hit, snippets = _find_pattern_hits(note_text, _MEDICATION_FOLLOWUP_PATTERNS)
            if hit:
                result["N2003"] = True
                evidence["N2003"] = snippets
                confidence["N2003"] = _confidence_from_entities(entities, snippets, base=0.79)

        if "N2005" in item_ids:
            hit, snippets = _find_pattern_hits(note_text, _MEDICATION_INTERVENTION_PATTERNS)
            if hit:
                result["N2005"] = True
                evidence["N2005"] = snippets
                confidence["N2005"] = _confidence_from_entities(entities, snippets, base=0.79)

        return result

    def _extract_treatment_items(
        self,
        note_text: str,
        entities: List[Dict[str, Any]],
        item_ids: set[str],
        confidence: Dict[str, float],
        evidence: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for item_id, patterns in self.treatment_patterns.items():
            if item_id not in item_ids:
                continue
            hit, snippets = _find_pattern_hits(note_text, patterns)
            if hit:
                result[item_id] = True
                evidence[item_id] = snippets
                confidence[item_id] = _confidence_from_entities(entities, snippets, base=0.76)
        return result

    def _load_auto_patterns(self, auto_pattern_path: Optional[str]) -> None:
        """Load optional pattern overrides generated from the MDS PDF."""
        path = Path(auto_pattern_path) if auto_pattern_path else _DEFAULT_AUTO_PATTERN_PATH
        if not path.exists():
            return

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Could not read auto pattern file %s: %s", path, exc)
            return

        patterns_raw = payload.get("patterns", {}) if isinstance(payload, dict) else {}
        labels_raw = payload.get("pdf_labels", payload.get("labels", {})) if isinstance(payload, dict) else {}
        if not isinstance(patterns_raw, dict):
            return

        self.diagnosis_patterns = _merge_pattern_maps(
            self.diagnosis_patterns,
            _filter_compatible_auto_patterns(
                schema=self.schema,
                section_id="I",
                auto_patterns=patterns_raw.get("I", {}),
                auto_labels=labels_raw.get("I", {}) if isinstance(labels_raw, dict) else {},
            ),
        )
        self.medication_patterns = _merge_pattern_maps(
            self.medication_patterns,
            _filter_compatible_auto_patterns(
                schema=self.schema,
                section_id="N",
                auto_patterns=patterns_raw.get("N", {}),
                auto_labels=labels_raw.get("N", {}) if isinstance(labels_raw, dict) else {},
            ),
        )
        self.treatment_patterns = _merge_pattern_maps(
            self.treatment_patterns,
            _filter_compatible_auto_patterns(
                schema=self.schema,
                section_id="O",
                auto_patterns=patterns_raw.get("O", {}),
                auto_labels=labels_raw.get("O", {}) if isinstance(labels_raw, dict) else {},
            ),
        )

        logger.info("Loaded auto patterns from %s", path)

    def _fill_i8000_from_entities(
        self,
        result: Dict[str, Any],
        entities: List[Dict[str, Any]],
        item_ids: set[str],
        confidence: Dict[str, float],
    ) -> None:
        if "I8000" not in item_ids:
            return

        if "I8000" in result:
            return

        candidates = []
        for ent in entities:
            label = str(ent.get("label", ""))
            if not any(token in label for token in ("DISEASE", "PROBLEM", "DISORDER", "CONDITION")):
                continue
            text = str(ent.get("text", "")).strip()
            if len(text) < 3:
                continue
            candidates.append(text)

        unique = _unique_preserve_order(candidates)
        if unique:
            result["I8000"] = ", ".join(unique[:8])
            confidence["I8000"] = min(0.88, 0.55 + 0.03 * len(unique))


def _infer_days_from_snippets(snippets: Sequence[str]) -> Optional[int]:
    """Infer day-count from snippets like 'for 5 days', 'x 7d', or '7/7'."""
    for snippet in snippets:
        candidates = [
            re.search(r"\b([1-7])\s*/\s*7\b", snippet),
            re.search(r"\b(?:for\s+)?([1-7])\s*(?:days?|d)\b", snippet, flags=re.IGNORECASE),
        ]
        for match in candidates:
            if match:
                return int(match.group(1))
    return None


def _merge_pattern_maps(
    base: Dict[str, Tuple[str, ...]],
    auto_raw: Any,
) -> Dict[str, Tuple[str, ...]]:
    """Merge base pattern map with auto-generated patterns while preserving order."""
    merged: Dict[str, Tuple[str, ...]] = dict(base)
    if not isinstance(auto_raw, dict):
        return merged

    for item_id, values in auto_raw.items():
        if not isinstance(item_id, str):
            continue
        base_values = list(merged.get(item_id, ()))
        incoming: List[str] = []

        if isinstance(values, list):
            incoming = [str(v).strip() for v in values if str(v).strip()]
        elif isinstance(values, str):
            if values.strip():
                incoming = [values.strip()]

        deduped = _unique_preserve_order([*base_values, *incoming])
        if deduped:
            merged[item_id] = tuple(deduped)

    return merged


def _schema_label_patterns(
    schema: MDSSchema,
    section_id: str,
    item_filter: Optional[Callable[[MDSItem], bool]] = None,
) -> Dict[str, Tuple[str, ...]]:
    """Build regex patterns from schema item labels for a section."""
    section = schema.get_section(section_id)
    if section is None:
        return {}

    out: Dict[str, Tuple[str, ...]] = {}
    for item in section.items:
        if item_filter is not None and not item_filter(item):
            continue
        terms = _label_terms(item.label)
        patterns = [_term_to_regex(term) for term in terms if term]
        deduped = _unique_preserve_order(patterns)
        if deduped:
            out[item.item_id] = tuple(deduped)
    return out


def _label_terms(label: str) -> List[str]:
    """Extract phrase candidates from a schema label (full phrase + aliases)."""
    text = (label or "").replace("—", "-").strip()
    if not text:
        return []

    terms: List[str] = [text]
    split_pattern = r",|\bor\b|/|\s[-:]\s"

    # Add parenthetical content, which often includes high-value abbreviations
    # such as CAD/COPD/PTSD.
    for content in re.findall(r"\(([^)]{1,80})\)", text):
        candidate = content.strip(" .;:,-")
        if candidate:
            terms.extend(re.split(split_pattern, candidate, flags=re.IGNORECASE))

    # Remove parenthetical clauses and split logical alternatives.
    no_parens = re.sub(r"\([^)]*\)", "", text)
    terms.extend(re.split(split_pattern, no_parens, flags=re.IGNORECASE))

    cleaned: List[str] = []
    for term in terms:
        t = re.sub(r"\s+", " ", term).strip(" .;:,-")
        if not t:
            continue
        if len(t) < 3:
            continue
        if t.lower() in {"none of the above", "none"}:
            continue
        cleaned.append(t)

    return _unique_preserve_order(cleaned)


def _term_to_regex(term: str) -> str:
    """Convert label term into a robust regex (supports whitespace/hyphen variation)."""
    # Ignore highly generic single-token terms from schema labels.
    words = re.findall(r"[A-Za-z]+", term.lower())
    if len(words) == 1 and words[0] in _GENERIC_SCHEMA_TOKENS:
        return ""

    escaped = re.escape(term)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\-", r"[-\s]?")
    return rf"\b{escaped}\b"


def _filter_compatible_auto_patterns(
    schema: MDSSchema,
    section_id: str,
    auto_patterns: Any,
    auto_labels: Any,
) -> Dict[str, Any]:
    """Keep only auto patterns whose PDF label is compatible with schema label."""
    if not isinstance(auto_patterns, dict):
        return {}
    labels = auto_labels if isinstance(auto_labels, dict) else {}
    section = schema.get_section(section_id)
    allowed_ids = {item.item_id for item in section.items} if section else set()

    filtered: Dict[str, Any] = {}
    for item_id, values in auto_patterns.items():
        if item_id not in allowed_ids:
            continue
        schema_item = schema.get_item(item_id)
        if schema_item is None:
            continue

        pdf_label = str(labels.get(item_id, "")).strip()
        if not pdf_label:
            continue
        if not _labels_compatible(schema_item.label, pdf_label):
            continue
        filtered[item_id] = values
    return filtered


def _labels_compatible(schema_label: str, pdf_label: str) -> bool:
    """Heuristic compatibility check between schema and PDF item labels."""
    def _tokens(text: str) -> set[str]:
        parts = re.findall(r"[A-Za-z]+", text.lower())
        stop = {
            "the", "and", "or", "of", "with", "without", "other", "last", "days",
            "received", "medications", "disease", "disorder",
        }
        return {part for part in parts if len(part) > 2 and part not in stop}

    s_tokens = _tokens(schema_label)
    p_tokens = _tokens(pdf_label)
    if not s_tokens or not p_tokens:
        return False

    overlap = len(s_tokens.intersection(p_tokens))
    ratio = overlap / max(1, min(len(s_tokens), len(p_tokens)))
    return ratio >= 0.5


def _find_pattern_hits(note_text: str, patterns: Sequence[str]) -> Tuple[bool, List[str]]:
    """Return whether any regex in *patterns* matched and compact evidence snippets."""
    snippets: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, note_text, flags=re.IGNORECASE):
            start = max(0, match.start() - 45)
            end = min(len(note_text), match.end() + 65)
            snippet = note_text[start:end].strip().replace("\n", " ")
            if snippet:
                snippets.append(snippet)
            if len(snippets) >= 3:
                break
        if len(snippets) >= 3:
            break
    return (len(snippets) > 0), _unique_preserve_order(snippets)


def _confidence_from_entities(
    entities: Sequence[Dict[str, Any]],
    snippets: Sequence[str],
    base: float,
) -> float:
    """Compute confidence from NER scores + lexical evidence count."""
    if not snippets:
        return max(0.0, min(1.0, base - 0.2))

    snippet_text = " ".join(snippets).lower()
    matched_scores = [
        float(ent.get("score", 0.0))
        for ent in entities
        if str(ent.get("text", "")).lower() in snippet_text
    ]
    avg_score = sum(matched_scores) / len(matched_scores) if matched_scores else 0.0
    evidence_boost = min(0.12, 0.03 * len(snippets))
    ner_boost = 0.15 * avg_score
    return max(0.0, min(0.99, base + evidence_boost + ner_boost))


def _unique_preserve_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        lower = normalized.lower()
        if lower in seen:
            continue
        seen.add(lower)
        out.append(normalized)
    return out
