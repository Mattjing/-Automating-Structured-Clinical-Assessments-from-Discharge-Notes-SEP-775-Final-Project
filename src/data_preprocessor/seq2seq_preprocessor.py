"""
Simplified preprocessor for the Seq2Seq encoder-decoder extractor — Option 1.

The full preprocessor (``preprocessor.py``) is designed for rule-based and
GPT-based pipelines.  It does six heavy steps: cleaning, abbreviation expansion,
evidence scoring, assertion detection per snippet, knowledge graph construction,
and conflict detection.  Most of that work is redundant for a seq2seq model:

* Abbreviation expansion — biomedical models (BioBART, ClinicalT5) natively
  understand clinical shorthand; expanding it risks paraphrasing artifacts.
* Evidence scoring / block ranking — the encoder reads the full note
  bidirectionally, so there is no need to pre-select the "best" paragraphs.
* Knowledge graph nodes and edges — the seq2seq model learns structure
  implicitly from (note → MDS JSON) fine-tuning pairs.

What this preprocessor retains
-------------------------------
* Text cleaning        — removes MIMIC de-identification placeholders and
                         administrative header lines that add noise without
                         contributing clinical information.
* Structured data      — formats ICD codes, prescriptions, and procedure rows
                         as a concise text block appended to the encoder input.
* Conflict annotation  — flags cases where a structured ICD code is contradicted
                         by a free-text negation.  The model cannot infer source
                         authority on its own; an explicit flag helps it decide.

Public API
----------
    build_seq2seq_input(note_text, sections, note_metadata, max_chars) -> str

This is the only function the Seq2SeqExtractor needs to call.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Re-use the cleaning and assertion logic from the full preprocessor —
# these are pure, stateless functions with no side effects.
from src.data_preprocessor.preprocessor import (
    clean_discharge_text,
    detect_assertion,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BLOCK_SPLIT = re.compile(r"\n\s*\n+")

# Stopwords filtered out during Jaccard overlap comparison
_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
    "with", "for", "of", "in", "on", "at", "to", "from", "by", "and",
    "or", "not", "this", "that", "it", "as", "no", "patient", "history",
    "without", "due", "after", "during", "given",
}

# Dataset name → MDS section heuristic
_DATASET_SECTION: Dict[str, str] = {
    "diagnos":    "I",
    "problem":    "I",
    "condition":  "I",
    "icd":        "I",
    "prescri":    "N",
    "med":        "N",
    "drug":       "N",
    "pharm":      "N",
    "procedure":  "O",
    "treatment":  "O",
    "therapy":    "O",
    "service":    "O",
}


def _infer_section(dataset_name: str, allowed: Set[str]) -> Optional[str]:
    lowered = dataset_name.lower()
    for keyword, section in _DATASET_SECTION.items():
        if keyword in lowered and section in allowed:
            return section
    return None


def _row_to_text(row: Dict[str, str], max_chars: int = 200) -> str:
    text = ", ".join(f"{k}={v}" for k, v in row.items() if v)
    return text[:max_chars].rstrip() + ("…" if len(text) > max_chars else "")


def _significant_tokens(text: str) -> Set[str]:
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _significant_tokens(a), _significant_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Structured data formatting
# ---------------------------------------------------------------------------

def _format_structured(
    note_metadata: Optional[Dict[str, Any]],
    sections: Sequence[str],
    max_rows: int = 40,
) -> str:
    """
    Format structured clinical data (ICD codes, Rx, procedures) as plain text.

    Returns an empty string when no structured data is present.
    """
    if not isinstance(note_metadata, dict):
        return ""
    structured = note_metadata.get("structured_data", {})
    if not isinstance(structured, dict) or not structured:
        return ""

    allowed = {s.upper() for s in sections}
    lines: List[str] = []
    count = 0

    for dataset_name, rows in structured.items():
        if not isinstance(rows, list):
            continue
        section = _infer_section(dataset_name, allowed)
        for row in rows:
            if count >= max_rows:
                break
            if not isinstance(row, dict):
                continue
            clean_row = {str(k): str(v).strip() for k, v in row.items() if str(v).strip()}
            text = _row_to_text(clean_row)
            if not text:
                continue
            label = f"[{section}]" if section else "[?]"
            lines.append(f"- {label} {dataset_name}: {text}")
            count += 1

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def _detect_conflicts(
    note_text: str,
    note_metadata: Optional[Dict[str, Any]],
    sections: Sequence[str],
    overlap_threshold: float = 0.25,
) -> List[str]:
    """
    Identify structured facts that are contradicted by free-text negations.

    Returns a list of human-readable conflict annotations to be appended to
    the encoder input so the model can apply source-authority reasoning.

    Only CONFIRMED-in-structured vs NEGATED-in-unstructured conflicts are
    flagged — UNCERTAIN vs CONFIRMED is clinically normal and not flagged.
    """
    if not isinstance(note_metadata, dict):
        return []
    structured = note_metadata.get("structured_data", {})
    if not isinstance(structured, dict):
        return []

    allowed = {s.upper() for s in sections}

    # Collect structured facts
    struct_facts: List[Tuple[str, str]] = []  # (dataset_name, fact_text)
    for dataset_name, rows in structured.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            clean_row = {str(k): str(v).strip() for k, v in row.items() if str(v).strip()}
            text = _row_to_text(clean_row)
            if text:
                struct_facts.append((dataset_name, text))

    if not struct_facts:
        return []

    # Collect negated sentences from the unstructured note
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", note_text) if s.strip()]
    negated_sentences = [s for s in sentences if detect_assertion(s) == "NEGATED"]

    if not negated_sentences:
        return []

    conflicts: List[str] = []
    for dataset_name, fact_text in struct_facts:
        for neg_sent in negated_sentences:
            if _jaccard(fact_text, neg_sent) >= overlap_threshold:
                conflicts.append(
                    f"- {dataset_name} (CONFIRMED in structured data) conflicts with "
                    f'note text "{neg_sent[:120]}" (NEGATED). '
                    f"Prefer structured — discharge codes are authoritative."
                )
                break  # one conflict annotation per structured fact is enough

    return conflicts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_seq2seq_input(
    note_text: str,
    sections: Sequence[str],
    note_metadata: Optional[Dict[str, Any]] = None,
    max_chars: int = 3500,
) -> str:
    """
    Build an encoder-ready input string for the Seq2Seq extractor.

    This is the sole entry point of this preprocessor.  The returned string is
    passed directly to the tokenizer as the encoder input.

    Steps
    -----
    1. Clean the note text (remove MIMIC placeholders and admin headers).
    2. Append a compact structured data block (ICD codes, Rx, procedures).
    3. Append conflict annotations where a structured fact is negated in the note.
    4. Truncate the combined string to ``max_chars`` characters (tail-first:
       the note body is trimmed first; the structured block is preserved).

    Parameters
    ----------
    note_text:
        Raw discharge note text.
    sections:
        MDS section IDs to extract (e.g. ``["I", "N", "O"]``).
    note_metadata:
        Optional metadata dict — may contain ``structured_data`` attached by
        ``MIMICDischargeLoader``.
    max_chars:
        Character budget for the combined encoder input.  At 4 chars/token on
        average, 3,500 chars ≈ 875 tokens — safe for both BART (1,024) and T5 (512).
        Increase for Longformer-ED models.

    Returns
    -------
    str
        Encoder-ready input string.

    Example output
    --------------
    ::

        Clinical note:
        Patient is a 78 yo male admitted for CHF exacerbation.
        Discharge diagnosis: Heart failure, AKI.
        Warfarin 5mg PO daily continued. Furosemide 40mg BID.
        PICC placed on HD3. BiPAP initiated overnight.
        Patient denies any history of atrial fibrillation.

        Structured data:
        - [I] diagnoses: icd_code=I50.9, description=Heart failure, unspecified
        - [I] diagnoses: icd_code=I48.0, description=Paroxysmal atrial fibrillation
        - [N] prescriptions: drug=Warfarin, dose=5 mg, route=PO, frequency=Daily

        Conflicts:
        - diagnoses (CONFIRMED in structured data) conflicts with note text
          "Patient denies any history of atrial fibrillation." (NEGATED).
          Prefer structured — discharge codes are authoritative.
    """
    cleaned = clean_discharge_text(note_text)

    structured_block = _format_structured(note_metadata, sections)
    conflict_lines   = _detect_conflicts(cleaned, note_metadata, sections)

    # Assemble suffix (structured + conflicts) — these are preserved in full
    suffix_parts: List[str] = []
    if structured_block:
        suffix_parts.append("Structured data:\n" + structured_block)
    if conflict_lines:
        suffix_parts.append("Conflicts:\n" + "\n".join(conflict_lines))
    suffix = ("\n\n" + "\n\n".join(suffix_parts)) if suffix_parts else ""

    # Truncate note body to fit within max_chars
    note_budget = max_chars - len(suffix)
    if note_budget < 200:
        # Suffix alone exceeds budget — trim suffix instead
        suffix = suffix[:max_chars - 200]
        note_budget = 200

    truncated_note = cleaned[:note_budget]
    if len(cleaned) > note_budget:
        # Trim at last sentence boundary to avoid mid-sentence cuts
        last_boundary = max(
            truncated_note.rfind(". "),
            truncated_note.rfind(".\n"),
        )
        if last_boundary > note_budget * 0.7:
            truncated_note = truncated_note[: last_boundary + 1]

    return ("Clinical note:\n" + truncated_note + suffix).strip()
