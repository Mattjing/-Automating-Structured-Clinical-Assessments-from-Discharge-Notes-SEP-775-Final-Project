"""
Utilities for cleaning and focusing discharge-note text before LLM extraction.

The preprocessing step removes administrative noise and de-identification
placeholders, then surfaces snippets most relevant to MDS sections I, N, and O.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple, cast

SectionTermMap = Dict[str, Tuple[str, ...]]
RankedBlock = Tuple[int, int, str]

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

_SECTION_HINTS = cast(SectionTermMap, {
    "I": (
        "diagnosis",
        "diagnoses",
        "discharge diagnosis",
        "discharge diagnoses",
        "principal diagnosis",
        "secondary diagnosis",
        "active problems",
        "problem list",
        "past medical history",
        "pmh",
        "comorbidity",
        "comorbidities",
        "history of",
        "chronic",
        "acute",
        "condition",
    ),
    "N": (
        "medication",
        "medications",
        "meds",
        "discharge meds",
        "home meds",
        "current meds",
        "medication list",
        "discharge medication",
        "dose",
        "dosage",
        "tablet",
        "capsule",
        "inject",
        "pharmacy",
        "insulin",
        "antibiotic",
        "anticoagulant",
        "warfarin",
        "heparin",
        "opioid",
    ),
    "O": (
        "procedure",
        "procedures",
        "procedure(s)",
        "operation",
        "surgery",
        "treatment",
        "treatments",
        "therapy",
        "oxygen",
        "dialysis",
        "iv",
        "intravenous",
        "transfusion",
        "chemotherapy",
        "radiation",
        "ventilator",
        "tracheostomy",
        "suction",
        "cpap",
        "bipap",
        "picc",
    ),
})

_SECTION_HEADER_HINTS = cast(SectionTermMap, {
    "I": (
        "discharge diagnosis",
        "discharge diagnoses",
        "diagnosis",
        "diagnoses",
        "active problems",
        "problem list",
        "past medical history",
        "pmh",
    ),
    "N": (
        "discharge medications",
        "discharge meds",
        "medications on discharge",
        "home medications",
        "home meds",
        "medication list",
        "medications",
        "meds",
    ),
    "O": (
        "procedures",
        "procedure",
        "treatments",
        "treatment",
        "therapy",
        "special treatments",
        "hospital course",
    ),
})

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_BLOCK_SPLIT_PATTERN = re.compile(r"\n\s*\n+")


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
    seen = set[str]()
    keywords: List[str] = []
    for section in sections:
        for word in _SECTION_HINTS.get(section.upper(), ()):  # unknown sections are ignored
            if word not in seen:
                seen.add(word)
                keywords.append(word)
    return keywords


def _split_into_blocks(text: str) -> List[str]:
    blocks = [block.strip() for block in _BLOCK_SPLIT_PATTERN.split(text) if block.strip()]
    return blocks or ([text.strip()] if text.strip() else [])


def _score_block(block: str, section: str) -> int:
    lowered = block.lower()
    score = 0

    for header in _SECTION_HEADER_HINTS.get(section, ()): 
        if header in lowered:
            score += 5

    for keyword in _SECTION_HINTS.get(section, ()): 
        score += lowered.count(keyword)

    return score


def _ranked_block_key(entry: RankedBlock) -> Tuple[int, int]:
    return (-entry[0], entry[1])


def _trim_block(block: str, section: str, max_sentences: int) -> str:
    header_hints = _SECTION_HEADER_HINTS.get(section, ())
    keywords = _SECTION_HINTS.get(section, ())
    sentences = [segment.strip() for segment in _SENTENCE_SPLIT_PATTERN.split(block) if segment.strip()]
    selected: List[str] = []

    for sentence in sentences:
        lowered = sentence.lower()
        if any(hint in lowered for hint in header_hints) or any(keyword in lowered for keyword in keywords):
            selected.append(sentence)
        if len(selected) >= max_sentences:
            break

    if not selected:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        selected = lines[:max_sentences]

    return " ".join(selected)


def extract_priority_snippets(text: str, sections: Sequence[str], max_sentences: int = 80) -> List[str]:
    """Return note snippets most likely to contain I/N/O evidence."""
    normalized_sections = [section.upper() for section in sections]
    keywords = _keywords_for_sections(normalized_sections)
    if not keywords or not text.strip():
        return []

    blocks = _split_into_blocks(text)
    snippets: List[str] = []
    seen = set[str]()
    remaining_sentences = max_sentences

    for section in normalized_sections:
        ranked_blocks: List[RankedBlock] = []
        for index, block in enumerate(blocks):
            score = _score_block(block, section)
            if score > 0:
                ranked_blocks.append((score, index, block))

        ranked_blocks.sort(key=_ranked_block_key)

        for _, _, block in ranked_blocks[:3]:
            if remaining_sentences <= 0:
                break

            snippet = _trim_block(block, section, max_sentences=min(remaining_sentences, 4))
            if not snippet:
                continue

            labelled_snippet = f"[{section}] {snippet}"
            if labelled_snippet in seen:
                continue

            seen.add(labelled_snippet)
            snippets.append(labelled_snippet)
            remaining_sentences -= max(1, len(_SENTENCE_SPLIT_PATTERN.split(snippet)))

    if snippets:
        return snippets

    fallback_sentences = [segment.strip() for segment in _SENTENCE_SPLIT_PATTERN.split(text) if segment.strip()]
    for sentence in fallback_sentences[:max_sentences]:
        low = sentence.lower()
        if any(keyword in low for keyword in keywords):
            snippets.append(sentence)

    return snippets


def build_extraction_context(
    note_text: str,
    sections: Sequence[str],
    max_clean_chars: int = 2500,
    max_snippet_chars: int = 4500,
) -> str:
    """
    Build focused context for the extractor.

    Returns a compact context block containing prioritized I/N/O evidence.
    """
    cleaned = clean_discharge_text(note_text)
    if not cleaned:
        return ""

    snippets = extract_priority_snippets(cleaned, sections)
    snippet_block = "\n".join(f"- {s}" for s in snippets)

    parts: List[str] = []
    parts.append("=== TARGET SECTIONS ===\n" + ", ".join(section.upper() for section in sections))

    if snippet_block:
        parts.append("=== PRIORITY EVIDENCE ===\n" + snippet_block[:max_snippet_chars])

    if not snippets or len(snippet_block) < 600:
        parts.append("=== SUPPORTING NOTE EXCERPT ===\n" + cleaned[:max_clean_chars])

    return "\n\n".join(parts)
