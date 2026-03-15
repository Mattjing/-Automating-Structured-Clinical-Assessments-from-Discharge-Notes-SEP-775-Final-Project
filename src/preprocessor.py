"""
Utilities for cleaning and focusing discharge-note text before LLM extraction.

The preprocessing step removes administrative noise and de-identification
placeholders, then surfaces snippets most relevant to MDS sections I, N, and O.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

_DEID_PATTERN = re.compile(r"\[\*\*.*?\*\*\]")
_UNDERSCORE_PATTERN = re.compile(r"_{2,}")
_MULTISPACE_PATTERN = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")

_ADMIN_PATTERNS = [
    re.compile(r"^name:\s*", re.IGNORECASE),
    re.compile(r"^unit\s*no:\s*", re.IGNORECASE),
    re.compile(r"^admission\s*date:\s*", re.IGNORECASE),
    re.compile(r"^discharge\s*date:\s*", re.IGNORECASE),
    re.compile(r"^date\s*of\s*birth:\s*", re.IGNORECASE),
    re.compile(r"^sex:\s*", re.IGNORECASE),
]

_SECTION_HINTS = {
    "I": (
        "diagnosis",
        "diagnoses",
        "active problems",
        "problem list",
        "past medical history",
        "pmh",
        "comorbidity",
    ),
    "N": (
        "medication",
        "medications",
        "meds",
        "discharge meds",
        "home meds",
        "pharmacy",
        "insulin",
        "antibiotic",
        "anticoagulant",
    ),
    "O": (
        "procedure",
        "procedures",
        "treatment",
        "therapy",
        "oxygen",
        "dialysis",
        "iv",
        "transfusion",
        "chemotherapy",
        "radiation",
        "ventilator",
        "tracheostomy",
        "suction",
    ),
}


def clean_discharge_text(text: str) -> str:
    """Normalize whitespace and remove low-value header artifacts."""
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _DEID_PATTERN.sub(" ", cleaned)
    cleaned = _UNDERSCORE_PATTERN.sub(" ", cleaned)

    kept_lines: List[str] = []
    for line in cleaned.split("\n"):
        line = _MULTISPACE_PATTERN.sub(" ", line).strip()
        if not line:
            kept_lines.append("")
            continue
        if any(p.search(line) for p in _ADMIN_PATTERNS):
            continue
        kept_lines.append(line)

    collapsed = "\n".join(kept_lines)
    collapsed = _MULTISPACE_PATTERN.sub(" ", collapsed)
    collapsed = _MULTI_NEWLINE_PATTERN.sub("\n\n", collapsed)
    return collapsed.strip()


def _keywords_for_sections(sections: Iterable[str]) -> List[str]:
    seen = set()
    keywords: List[str] = []
    for section in sections:
        for word in _SECTION_HINTS.get(section.upper(), ()):  # unknown sections are ignored
            if word not in seen:
                seen.add(word)
                keywords.append(word)
    return keywords


def extract_priority_snippets(text: str, sections: Sequence[str], max_sentences: int = 80) -> List[str]:
    """Return note sentences likely to contain I/N/O evidence."""
    keywords = _keywords_for_sections(sections)
    if not keywords:
        return []

    sentence_regex = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_regex.split(text)

    snippets: List[str] = []
    for sentence in sentences:
        low = sentence.lower()
        if any(k in low for k in keywords):
            stripped = sentence.strip()
            if stripped:
                snippets.append(stripped)
        if len(snippets) >= max_sentences:
            break

    return snippets


def build_extraction_context(
    note_text: str,
    sections: Sequence[str],
    max_clean_chars: int = 12000,
    max_snippet_chars: int = 4000,
) -> str:
    """
    Build focused context for the extractor.

    Returns a composite block containing prioritized snippets plus cleaned text.
    """
    cleaned = clean_discharge_text(note_text)
    if not cleaned:
        return ""

    snippets = extract_priority_snippets(cleaned, sections)
    snippet_block = "\n".join(f"- {s}" for s in snippets)

    parts: List[str] = []
    if snippet_block:
        parts.append("=== PRIORITY SNIPPETS ===\n" + snippet_block[:max_snippet_chars])

    parts.append("=== CLEANED NOTE ===\n" + cleaned[:max_clean_chars])
    return "\n\n".join(parts)
