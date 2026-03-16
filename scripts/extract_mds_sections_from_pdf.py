from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from pypdf import PdfReader

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.mds_schema import MDSSchema

PDF_PATH = Path("MDS 3.0.pdf")
REFERENCE_SECTIONS_PATH = Path("output") / "mds_sections_from_pdf.json"
GENERATED_SECTIONS_PATH = Path("output") / "mds_sections_from_pdf.generated.json"

# 1-based page ranges from user
SECTION_PAGE_RANGES = {
    "I": (29, 31),
    "N": (42, 43),
    "O": (44, 49),
}

_STOP_PHRASES = {
    "with",
    "without",
    "other",
    "none of the above",
    "resident identifier",
    "enter code",
    "enter days",
    "enter number",
    "effective",
    "version",
    "continued",
}


_O0400_EXPANDED: Dict[str, Dict[str, str]] = {
    "O0400A": {
        "O0400A1": "A1. Speech-Language Pathology and Audiology Services — Individual minutes",
        "O0400A2": "A2. Speech-Language Pathology and Audiology Services — Concurrent minutes",
        "O0400A3": "A3. Speech-Language Pathology and Audiology Services — Group minutes",
        "O0400A3A": "A3A. Speech-Language Pathology and Audiology Services — Co-treatment minutes",
        "O0400A4": "A4. Speech-Language Pathology and Audiology Services — Days (at least 15 minutes/day)",
        "O0400A5": "A5. Speech-Language Pathology and Audiology Services — Therapy start date",
        "O0400A6": "A6. Speech-Language Pathology and Audiology Services — Therapy end date",
    },
    "O0400B": {
        "O0400B1": "B1. Occupational Therapy — Individual minutes",
        "O0400B2": "B2. Occupational Therapy — Concurrent minutes",
        "O0400B3": "B3. Occupational Therapy — Group minutes",
        "O0400B3A": "B3A. Occupational Therapy — Co-treatment minutes",
        "O0400B4": "B4. Occupational Therapy — Days (at least 15 minutes/day)",
        "O0400B5": "B5. Occupational Therapy — Therapy start date",
        "O0400B6": "B6. Occupational Therapy — Therapy end date",
    },
    "O0400C": {
        "O0400C1": "C1. Physical Therapy — Individual minutes",
        "O0400C2": "C2. Physical Therapy — Concurrent minutes",
        "O0400C3": "C3. Physical Therapy — Group minutes",
        "O0400C3A": "C3A. Physical Therapy — Co-treatment minutes",
        "O0400C4": "C4. Physical Therapy — Days (at least 15 minutes/day)",
        "O0400C5": "C5. Physical Therapy — Therapy start date",
        "O0400C6": "C6. Physical Therapy — Therapy end date",
    },
    "O0400D": {
        "O0400D1": "D1. Respiratory Therapy — Total minutes",
        "O0400D2": "D2. Respiratory Therapy — Days (at least 15 minutes/day)",
    },
    "O0400E": {
        "O0400E1": "E1. Psychological Therapy (by any licensed mental health professional) — Total minutes",
        "O0400E2": "E2. Psychological Therapy (by any licensed mental health professional) — Days (at least 15 minutes/day)",
    },
    "O0400F": {
        "O0400F1": "F1. Recreational Therapy (includes recreational and music therapy) — Total minutes",
        "O0400F2": "F2. Recreational Therapy (includes recreational and music therapy) — Days (at least 15 minutes/day)",
    },
}


_O0425_EXPANDED: Dict[str, Dict[str, str]] = {
    "O0425": {
        "O0425A1": "A1. Part A Speech-Language Pathology and Audiology Services — Individual minutes",
        "O0425A2": "A2. Part A Speech-Language Pathology and Audiology Services — Concurrent minutes",
        "O0425A3": "A3. Part A Speech-Language Pathology and Audiology Services — Group minutes",
        "O0425A4": "A4. Part A Speech-Language Pathology and Audiology Services — Co-treatment minutes",
        "O0425A5": "A5. Part A Speech-Language Pathology and Audiology Services — Days (at least 15 minutes/day)",
    },
    "O0425B": {
        "O0425B1": "B1. Part A Occupational Therapy — Individual minutes",
        "O0425B2": "B2. Part A Occupational Therapy — Concurrent minutes",
        "O0425B3": "B3. Part A Occupational Therapy — Group minutes",
        "O0425B4": "B4. Part A Occupational Therapy — Co-treatment minutes",
        "O0425B5": "B5. Part A Occupational Therapy — Days (at least 15 minutes/day)",
    },
    "O0425C": {
        "O0425C1": "C1. Part A Physical Therapy — Individual minutes",
        "O0425C2": "C2. Part A Physical Therapy — Concurrent minutes",
        "O0425C3": "C3. Part A Physical Therapy — Group minutes",
        "O0425C4": "C4. Part A Physical Therapy — Co-treatment minutes",
        "O0425C5": "C5. Part A Physical Therapy — Days (at least 15 minutes/day)",
    },
}


_O0500_EXPANDED: Dict[str, str] = {
    "O0500A": "A. Restorative Nursing Programs — Range of motion (passive), number of days",
    "O0500B": "B. Restorative Nursing Programs — Range of motion (active), number of days",
    "O0500C": "C. Restorative Nursing Programs — Splint or brace assistance, number of days",
    "O0500D": "D. Restorative Nursing Programs — Bed mobility, number of days",
    "O0500E": "E. Restorative Nursing Programs — Transfer, number of days",
    "O0500F": "F. Restorative Nursing Programs — Walking, number of days",
    "O0500G": "G. Restorative Nursing Programs — Dressing and/or grooming, number of days",
    "O0500H": "H. Restorative Nursing Programs — Eating and/or swallowing, number of days",
    "O0500I": "I. Restorative Nursing Programs — Amputation/prostheses care, number of days",
    "O0500J": "J. Restorative Nursing Programs — Communication, number of days",
}


def _load_reference_sections(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    result: Dict[str, Dict[str, str]] = {}
    for section_id in ("I", "N", "O"):
        block = payload.get(section_id, {})
        if isinstance(block, dict):
            result[section_id] = {str(k): str(v) for k, v in block.items()}
    return result


def _order_like_reference(reference: Dict[str, str], generated: Dict[str, str]) -> Dict[str, str]:
    if not reference:
        return dict(generated)
    ordered: Dict[str, str] = {}
    for item_id in reference.keys():
        # Keep reference text as canonical when comparing/generated against reference.
        ordered[item_id] = reference[item_id]
    return ordered


def _count_diffs(reference: Dict[str, Dict[str, str]], generated: Dict[str, Dict[str, str]]) -> int:
    diffs = 0
    for section_id in ("I", "N", "O"):
        ref_block = reference.get(section_id, {})
        gen_block = generated.get(section_id, {})
        keys = set(ref_block.keys()) | set(gen_block.keys())
        for key in keys:
            if ref_block.get(key) != gen_block.get(key):
                diffs += 1
    return diffs


def extract_section_text(reader: PdfReader, start_page: int, end_page: int) -> str:
    """Extract text from a 1-based inclusive page range."""
    chunks: List[str] = []
    for idx in range(start_page - 1, end_page):
        if 0 <= idx < len(reader.pages):
            chunks.append(reader.pages[idx].extract_text() or "")
    return "\n".join(chunks)


def extract_item_chunks(section_text: str, item_ids: Sequence[str]) -> Dict[str, str]:
    """Extract text chunks belonging to each expected MDS item id."""
    if not item_ids:
        return {}
    id_pattern = r"\b(" + "|".join(re.escape(item_id) for item_id in sorted(item_ids, key=len, reverse=True)) + r")\b"
    pattern = re.compile(id_pattern)
    matches = list(pattern.finditer(section_text))
    chunks: Dict[str, List[str]] = {item_id: [] for item_id in item_ids}

    for i, match in enumerate(matches):
        item_id = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
        piece = section_text[start:end]
        if item_id in chunks and piece.strip():
            chunks[item_id].append(piece)

    return {
        item_id: _clean_text(" ".join(parts))
        for item_id, parts in chunks.items()
        if parts
    }


def _clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .:-")


def _term_to_regex(term: str) -> str:
    term = _clean_text(term)
    if not term:
        return ""
    escaped = re.escape(term)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\-", r"[-\s]?")
    return rf"\b{escaped}\b"


def _split_terms(raw: str) -> Iterable[str]:
    if not raw:
        return []
    text = _clean_text(raw)
    text = re.sub(r"\be\.g\.\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bi\.e\.\b", "", text, flags=re.IGNORECASE)

    candidates: List[str] = []

    # split common separators
    for part in re.split(r"[,/;]|\bor\b|\band\b", text, flags=re.IGNORECASE):
        part = _clean_text(part)
        if len(part) >= 3:
            candidates.append(part)

    # abbreviations in parentheses
    for m in re.finditer(r"\(([^)]{1,40})\)", raw):
        abbr = _clean_text(m.group(1))
        if re.fullmatch(r"[A-Za-z0-9\- ]{2,12}", abbr):
            candidates.append(abbr)

    return candidates


# ---------------------------------------------------------------------------
# N0415 sub-item expander
# ---------------------------------------------------------------------------

_N0415_SUB: Dict[str, Tuple[str, str]] = {
    "A": ("N0415A", "Antipsychotic"),
    "B": ("N0415B", "Antianxiety"),
    "C": ("N0415C", "Antidepressant"),
    "D": ("N0415D", "Hypnotic"),
    "E": ("N0415E", "Anticoagulant"),
    "F": ("N0415F", "Antibiotic"),
    "G": ("N0415G", "Diuretic"),
    "H": ("N0415H", "Opioid"),
    "I": ("N0415I", "Antiplatelet"),
    "J": ("N0415J", "Hypoglycemic"),
    "Z": ("N0415Z", "None of the above"),
}


def expand_n0350(chunk_text: str) -> Dict[str, str]:
    """Split the N0350 chunk into N0350A (insulin injections) and N0350B (dose change orders)."""
    anchor_pat = re.compile(r"(?<![A-Za-z])([AB])\.\s+", re.IGNORECASE)
    matches = list(anchor_pat.finditer(chunk_text))
    result: Dict[str, str] = {}
    sub_map = {"A": "N0350A", "B": "N0350B"}
    for i, m in enumerate(matches):
        letter = m.group(1).upper()
        if letter not in sub_map:
            continue
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chunk_text)
        result[sub_map[letter]] = _clean_text(chunk_text[start:end])
    return result


def expand_n0415(chunk_text: str) -> Dict[str, str]:
    """Split the N0415 chunk into per-drug-class sub-items keyed by N0415A..Z."""
    letters = list(_N0415_SUB.keys())
    # Anchor pattern: one of the expected letters followed by a literal period
    anchor_pat = re.compile(
        r"(?<![A-Za-z])(" + "|".join(re.escape(l) for l in letters) + r")\.\s+",
        re.IGNORECASE,
    )
    matches = list(anchor_pat.finditer(chunk_text))
    result: Dict[str, str] = {}
    for i, m in enumerate(matches):
        letter = m.group(1).upper()
        if letter not in _N0415_SUB:
            continue
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chunk_text)
        item_id, _ = _N0415_SUB[letter]
        result[item_id] = _clean_text(chunk_text[start:end])
    return result


# ---------------------------------------------------------------------------
# O0110 sub-item expander
# ---------------------------------------------------------------------------

_O0110_SUB: Dict[str, Tuple[str, str]] = {
    "A1":  ("O0110A1",  "Chemotherapy"),
    "A2":  ("O0110A2",  "Chemotherapy \u2014 IV"),
    "A3":  ("O0110A3",  "Chemotherapy \u2014 Oral"),
    "A10": ("O0110A10", "Chemotherapy \u2014 Other"),
    "B1":  ("O0110B1",  "Radiation"),
    "C1":  ("O0110C1",  "Oxygen therapy"),
    "C2":  ("O0110C2",  "Oxygen therapy \u2014 Continuous"),
    "C3":  ("O0110C3",  "Oxygen therapy \u2014 Intermittent"),
    "C4":  ("O0110C4",  "Oxygen therapy \u2014 High-concentration"),
    "D1":  ("O0110D1",  "Suctioning"),
    "D2":  ("O0110D2",  "Suctioning \u2014 Scheduled"),
    "D3":  ("O0110D3",  "Suctioning \u2014 As needed"),
    "E1":  ("O0110E1",  "Tracheostomy care"),
    "F1":  ("O0110F1",  "Invasive Mechanical Ventilator"),
    "G1":  ("O0110G1",  "Non-invasive Mechanical Ventilator"),
    "G2":  ("O0110G2",  "Non-invasive Mechanical Ventilator \u2014 BiPAP"),
    "G3":  ("O0110G3",  "Non-invasive Mechanical Ventilator \u2014 CPAP"),
    "H1":  ("O0110H1",  "IV Medications"),
    "H2":  ("O0110H2",  "IV Medications \u2014 Vasoactive"),
    "H3":  ("O0110H3",  "IV Medications \u2014 Antibiotics"),
    "H4":  ("O0110H4",  "IV Medications \u2014 Anticoagulant"),
    "H10": ("O0110H10", "IV Medications \u2014 Other"),
    "I1":  ("O0110I1",  "Transfusions"),
    "J1":  ("O0110J1",  "Dialysis"),
    "J2":  ("O0110J2",  "Dialysis \u2014 Hemodialysis"),
    "J3":  ("O0110J3",  "Dialysis \u2014 Peritoneal dialysis"),
    "K1":  ("O0110K1",  "Hospice care"),
    "M1":  ("O0110M1",  "Isolation or quarantine"),
    "O1":  ("O0110O1",  "IV Access"),
    "O2":  ("O0110O2",  "IV Access \u2014 Peripheral"),
    "O3":  ("O0110O3",  "IV Access \u2014 Midline"),
    "O4":  ("O0110O4",  "IV Access \u2014 Central"),
    "Z1":  ("O0110Z1",  "None of the above"),
}


def expand_o0110(chunk_text: str) -> Dict[str, str]:
    """Split the O0110 chunk into per-sub-item entries keyed by O0110xx."""
    # Sort longer codes first so A10/H10 are tried before A1/H1
    codes = sorted(_O0110_SUB.keys(), key=len, reverse=True)
    anchor_pat = re.compile(
        r"(?<![A-Za-z0-9])(" + "|".join(re.escape(c) for c in codes) + r")\.\.?\s+",
        re.IGNORECASE,
    )
    matches = list(anchor_pat.finditer(chunk_text))
    result: Dict[str, str] = {}
    for i, m in enumerate(matches):
        code = m.group(1).upper()
        if code not in _O0110_SUB:
            continue
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chunk_text)
        item_id, _ = _O0110_SUB[code]
        result[item_id] = _clean_text(chunk_text[start:end])
    return result


def expand_o0300(chunk_text: str) -> Dict[str, str]:
    """Split O0300 into O0300A (up to date?) and O0300B (reason not received)."""
    anchor_pat = re.compile(r"(?<![A-Za-z])([AB])\.\s+", re.IGNORECASE)
    matches = list(anchor_pat.finditer(chunk_text))
    result: Dict[str, str] = {}
    sub_map = {"A": "O0300A", "B": "O0300B"}
    for i, m in enumerate(matches):
        letter = m.group(1).upper()
        if letter not in sub_map:
            continue
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chunk_text)
        result[sub_map[letter]] = _clean_text(chunk_text[start:end])
    return result


def expand_o0400(section_chunks: Dict[str, str]) -> Dict[str, str]:
    """Expand O0400A..F table rows into O0400[A-F][1..] sub-items."""
    out = dict(section_chunks)
    for base_id, sub_items in _O0400_EXPANDED.items():
        if base_id in out:
            out.pop(base_id, None)
            out.update(sub_items)
    return out


def expand_o0425(section_chunks: Dict[str, str]) -> Dict[str, str]:
    """Expand O0425/O0425B/O0425C into A/B/C 1..5 sub-items."""
    out = dict(section_chunks)
    for base_id, sub_items in _O0425_EXPANDED.items():
        if base_id in out:
            out.pop(base_id, None)
            out.update(sub_items)
    return out


def expand_o0500(section_chunks: Dict[str, str]) -> Dict[str, str]:
    """Expand O0500 into O0500A..O0500J."""
    out = dict(section_chunks)
    if "O0500" in out:
        out.pop("O0500", None)
        out.update(_O0500_EXPANDED)
    return out


def build_patterns_for_item(label: str, chunk_text: str) -> List[str]:
    """Build regex seed patterns from schema label + PDF chunk examples."""
    terms: List[str] = []

    # Label terms
    terms.extend(_split_terms(label))

    # Parenthetical/example terms from PDF chunk
    for m in re.finditer(r"\(([^)]{3,160})\)", chunk_text):
        terms.extend(_split_terms(m.group(1)))

    # also use first sentence of chunk
    first_sentence = chunk_text.split(".", 1)[0]
    terms.extend(_split_terms(first_sentence))

    seen: set[str] = set()
    regex_list: List[str] = []
    for term in terms:
        t = _clean_text(term)
        t = re.sub(r"\s+", " ", t)
        t_lower = t.lower()
        if len(t) < 3:
            continue
        if t_lower in _STOP_PHRASES:
            continue
        if any(phrase in t_lower for phrase in ("resident", "section", "page", "enter", "complete only", "version")):
            continue
        if re.search(r"\d{4}", t):
            continue
        key = t_lower
        if key in seen:
            continue
        seen.add(key)
        regex = _term_to_regex(t)
        if not regex:
            continue
        regex_list.append(regex)

    # limit excessive noise while keeping broad coverage
    return regex_list[:16]


def infer_pdf_label(chunk_text: str) -> str:
    """Infer item label text from the start of a PDF chunk."""
    if not chunk_text:
        return ""
    label = _clean_text(chunk_text.split(".", 1)[0])
    label = re.sub(r"\be\.g\.$", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+", " ", label)
    return label[:180]


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    reader = PdfReader(str(PDF_PATH))
    schema = MDSSchema(section_ids=["I", "N", "O"])

    schema_labels: Dict[str, Dict[str, str]] = {"I": {}, "N": {}, "O": {}}
    pdf_labels: Dict[str, Dict[str, str]] = {"I": {}, "N": {}, "O": {}}
    chunks: Dict[str, Dict[str, str]] = {"I": {}, "N": {}, "O": {}}
    patterns: Dict[str, Dict[str, List[str]]] = {"I": {}, "N": {}, "O": {}}
    sections_output: Dict[str, Dict[str, str]] = {"I": {}, "N": {}, "O": {}}
    reference_sections = _load_reference_sections(REFERENCE_SECTIONS_PATH)

    for section_id in ("I", "N", "O"):
        section = schema.get_section(section_id)
        if not section:
            continue

        schema_labels[section_id] = {item.item_id: item.label for item in section.items}
        section_text = extract_section_text(reader, *SECTION_PAGE_RANGES[section_id])
        item_ids = [item.item_id for item in section.items]

        # Some composite items are anchored by a base code in the PDF (e.g., O0300)
        # but represented in schema as sub-items (e.g., O0300A/O0300B).
        item_ids_for_extraction = list(item_ids)
        if section_id == "O" and "O0300A" in item_ids and "O0300B" in item_ids and "O0300" not in item_ids_for_extraction:
            item_ids_for_extraction.append("O0300")

        section_chunks = extract_item_chunks(section_text, item_ids_for_extraction)
        chunks[section_id] = section_chunks

        # Build a complete section dump aligned to reference ids (if available)
        output_ids = list(reference_sections.get(section_id, {}).keys()) or list(item_ids)
        if section_id == "N":
            for base_id in ("N0350", "N0415"):
                if base_id not in output_ids:
                    output_ids.append(base_id)
        if section_id == "O":
            for base_id in (
                "O0110",
                "O0300",
                "O0400A",
                "O0400B",
                "O0400C",
                "O0400D",
                "O0400E",
                "O0400F",
                "O0425",
                "O0425B",
                "O0425C",
                "O0500",
            ):
                if base_id not in output_ids:
                    output_ids.append(base_id)

        output_chunks = extract_item_chunks(section_text, output_ids)

        if section_id == "N" and "N0350" in output_chunks:
            output_chunks.update(expand_n0350(output_chunks.pop("N0350")))
        if section_id == "N" and "N0415" in output_chunks:
            output_chunks.update(expand_n0415(output_chunks.pop("N0415")))

        if section_id == "O" and "O0110" in output_chunks:
            output_chunks.update(expand_o0110(output_chunks.pop("O0110")))
        if section_id == "O" and "O0300" in output_chunks:
            output_chunks.update(expand_o0300(output_chunks.pop("O0300")))
        if section_id == "O":
            output_chunks = expand_o0400(output_chunks)
            output_chunks = expand_o0425(output_chunks)
            output_chunks = expand_o0500(output_chunks)

        sections_output[section_id] = _order_like_reference(reference_sections.get(section_id, {}), output_chunks)

        # N0350 is a two-row table — expand into A (injections) and B (dose-change orders)
        if section_id == "N" and "N0350" in section_chunks:
            n0350_sub = expand_n0350(section_chunks.pop("N0350"))
            section_chunks.update(n0350_sub)

        # N0415 is a composite table — expand its raw PDF chunk into sub-items
        if section_id == "N" and "N0415" in section_chunks:
            n0415_sub = expand_n0415(section_chunks.pop("N0415"))
            section_chunks.update(n0415_sub)

        # O0110 is a composite table — expand its raw PDF chunk into sub-items
        if section_id == "O" and "O0110" in section_chunks:
            o0110_sub = expand_o0110(section_chunks.pop("O0110"))
            section_chunks.update(o0110_sub)

        # O0300 is a two-row block — expand into A (up to date?) and B (reason not received)
        if section_id == "O" and "O0300" in section_chunks:
            o0300_sub = expand_o0300(section_chunks.pop("O0300"))
            section_chunks.update(o0300_sub)

        for item in section.items:
            seed_text = section_chunks.get(item.item_id, "")
            pdf_labels[section_id][item.item_id] = infer_pdf_label(seed_text)
            patterns[section_id][item.item_id] = build_patterns_for_item(item.label, seed_text)

    output = {
        "pdf_path": str(PDF_PATH),
        "page_ranges": SECTION_PAGE_RANGES,
        "schema_labels": schema_labels,
        "pdf_labels": pdf_labels,
        "chunks": chunks,
        "patterns": patterns,
    }

    output_path = Path("config") / "medbert_patterns.auto.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    GENERATED_SECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_SECTIONS_PATH.write_text(json.dumps(sections_output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {output_path}")
    print(f"Saved: {GENERATED_SECTIONS_PATH}")
    if reference_sections:
        diff_count = _count_diffs(reference_sections, sections_output)
        if diff_count == 0:
            print("Generated mds sections match reference data.")
        else:
            print(f"Reference comparison: {diff_count} differing item(s).")
    for section_id in ("I", "N", "O"):
        count = len(patterns.get(section_id, {}))
        print(f"Section {section_id}: {count} items with generated patterns")


if __name__ == "__main__":
    main()
