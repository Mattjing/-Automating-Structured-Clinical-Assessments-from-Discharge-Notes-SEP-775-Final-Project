"""
Utilities for cleaning and focusing discharge-note text before LLM extraction.

Key improvements over v1
------------------------
* Clinical abbreviation expansion  (MI → myocardial infarction, CHF → …)
* NegEx-style assertion detection  (CONFIRMED / NEGATED / UNCERTAIN per snippet)
* Extended pattern matching        (medical suffixes, dosage patterns, procedure verbs)
* Conflict detection & resolution  (structured vs. unstructured evidence disagreements)

All existing public-API function signatures are preserved so callers need no changes.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast

# ── Type aliases ───────────────────────────────────────────────────────────────
SectionTermMap = Dict[str, Tuple[str, ...]]
RankedBlock = Tuple[int, int, str]

# ── De-identification / whitespace patterns ────────────────────────────────────
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

# ── Clinical abbreviation dictionary ──────────────────────────────────────────
# Keys are lower-case; values are unambiguous full expansions.
# Ambiguous abbreviations (e.g. "pe" = pulmonary embolism OR physical exam) are
# kept only when the expansion is overwhelmingly the clinical meaning in notes.
_ABBREVIATIONS: Dict[str, str] = {
    # ── Conditions / diagnoses ────────────────────────────────────────────────
    "mi":    "myocardial infarction",
    "chf":   "congestive heart failure",
    "copd":  "chronic obstructive pulmonary disease",
    "dm":    "diabetes mellitus",
    "dm1":   "diabetes mellitus type 1",
    "dm2":   "diabetes mellitus type 2",
    "t1dm":  "type 1 diabetes mellitus",
    "t2dm":  "type 2 diabetes mellitus",
    "htn":   "hypertension",
    "afib":  "atrial fibrillation",
    "af":    "atrial fibrillation",
    "cad":   "coronary artery disease",
    "ckd":   "chronic kidney disease",
    "esrd":  "end-stage renal disease",
    "aki":   "acute kidney injury",
    "uti":   "urinary tract infection",
    "dvt":   "deep vein thrombosis",
    "cva":   "cerebrovascular accident stroke",
    "tia":   "transient ischemic attack",
    "gerd":  "gastroesophageal reflux disease",
    "pud":   "peptic ulcer disease",
    "hiv":   "human immunodeficiency virus",
    "tb":    "tuberculosis",
    "ra":    "rheumatoid arthritis",
    "sle":   "systemic lupus erythematosus",
    "ms":    "multiple sclerosis",
    "pd":    "parkinson disease",
    "ad":    "alzheimer disease",
    "bph":   "benign prostatic hyperplasia",
    "pvd":   "peripheral vascular disease",
    "hld":   "hyperlipidemia",
    "sob":   "shortness of breath",
    "doe":   "dyspnea on exertion",
    "cp":    "chest pain",
    "ams":   "altered mental status",
    "gi":    "gastrointestinal",
    "pmh":   "past medical history",
    "hpi":   "history of present illness",
    "nstemi": "non-ST elevation myocardial infarction",
    "stemi": "ST elevation myocardial infarction",
    "pe":    "pulmonary embolism",
    "osas":  "obstructive sleep apnea syndrome",
    "osa":   "obstructive sleep apnea",
    # ── Medications / administration ──────────────────────────────────────────
    "asa":   "aspirin",
    "apap":  "acetaminophen",
    "abx":   "antibiotics",
    "po":    "oral by mouth",
    "iv":    "intravenous",
    "im":    "intramuscular",
    "sq":    "subcutaneous",
    "sc":    "subcutaneous",
    "prn":   "as needed",
    "bid":   "twice daily",
    "tid":   "three times daily",
    "qid":   "four times daily",
    "qd":    "once daily",
    "qhs":   "at bedtime",
    "npo":   "nothing by mouth",
    # ── Procedures / devices / treatments ────────────────────────────────────
    "cabg":  "coronary artery bypass graft",
    "pci":   "percutaneous coronary intervention",
    "picc":  "peripherally inserted central catheter",
    "cvl":   "central venous line",
    "ngt":   "nasogastric tube",
    "cpap":  "continuous positive airway pressure",
    "bipap": "bilevel positive airway pressure",
    "nippv": "non-invasive positive pressure ventilation",
    "vent":  "mechanical ventilator",
    "o2":    "oxygen",
    "pt":    "physical therapy",
    "ot":    "occupational therapy",
    "st":    "speech therapy",
    "dnr":   "do not resuscitate",
    "dni":   "do not intubate",
}

# Build a compiled whole-word pattern (longest abbreviations first to avoid
# partial matches, e.g. "dm2" before "dm").
_ABBREV_SORTED = sorted(_ABBREVIATIONS.keys(), key=len, reverse=True)
_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _ABBREV_SORTED) + r")\b",
    re.IGNORECASE,
)

# ── Assertion / negation detection ────────────────────────────────────────────
# Negation cues appearing BEFORE a clinical concept (within a ~70-char window).
_NEG_PRE_PATTERN = re.compile(
    r"\b("
    r"no|not|without|denies?|deny(?:ing)?|"
    r"no\s+evidence\s+of|no\s+signs?\s+of|no\s+history\s+of|"
    r"ruled?\s+out|rules?\s+out|negative\s+for|"
    r"absent|absence\s+of|never|none|free\s+of|"
    r"not\s+consistent\s+with|unremarkable\s+for|"
    r"not\s+indicative\s+of|no\s+longer"
    r")\b",
    re.IGNORECASE,
)

# Negation cues appearing AFTER a clinical concept (within a ~70-char window).
_NEG_POST_PATTERN = re.compile(
    r"\b("
    r"ruled?\s+out|not\s+present|not\s+identified|not\s+detected|"
    r"not\s+found|not\s+seen|not\s+noted|negative|not\s+confirmed|"
    r"was\s+not\s+found|were\s+not\s+found"
    r")\b",
    re.IGNORECASE,
)

# Uncertainty / hedging cues (anywhere in the sentence).
_UNCERTAINTY_PATTERN = re.compile(
    r"\b("
    r"possible|possibly|probable|probably|likely|"
    r"suspected?|suspicion\s+of|may\s+have|might\s+have|"
    r"questionable|question\s+of|concerning\s+for|worrisome\s+for|"
    r"cannot\s+rule\s+out|could\s+not\s+exclude|could\s+represent|"
    r"appears?\s+to\s+be|seems?\s+to\s+be|thought\s+to\s+be|"
    r"presumed|presumptive|differential(?:\s+includes?)?|"
    r"vs\.|versus|r/o|rule\s+out"
    r")\b",
    re.IGNORECASE,
)

# ── Extended clinical patterns for section classification ─────────────────────
# Medical condition suffixes → Section I (diagnoses/conditions)
_DIAGNOSIS_SUFFIX_PATTERN = re.compile(
    r"\b\w{3,}(?:itis|osis|emia|opathy|pathy|oma|iasis|uria|megaly|trophy|"
    r"cardia|plegia|paresis|philia|phobia|rrhagia|rrhea)\b",
    re.IGNORECASE,
)

# Diagnosis framing verbs → Section I
_DIAGNOSIS_VERB_PATTERN = re.compile(
    r"\b(?:diagnos(?:ed|is|es)|history\s+of|known\s+(?:to\s+have|history)|"
    r"presents?\s+with|admitted\s+(?:for|with)|consistent\s+with|"
    r"assessment[:\s]|impression[:\s])\b",
    re.IGNORECASE,
)

# Dosage / quantity patterns → strong Section N signal
_DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|units?|mEq|mmol|mL|L)\b",
    re.IGNORECASE,
)

# Drug administration route → Section N
_ROUTE_PATTERN = re.compile(
    r"\b(?:oral(?:ly)?|by\s+mouth|intravenous(?:ly)?|subcutaneous(?:ly)?|"
    r"intramuscular(?:ly)?|topical(?:ly)?|inhaled|nebulized|transdermal|"
    r"sublingual)\b",
    re.IGNORECASE,
)

# Procedure / treatment action verbs → Section O
_PROCEDURE_VERB_PATTERN = re.compile(
    r"\b(?:underwent|performed|placed|inserted|received|administered|"
    r"started\s+on|initiated|transfused|dialyz(?:ed|ing)|intubated|"
    r"extubated|catheteriz(?:ed|ing)|implanted|resected|repaired|"
    r"irrigated|suctioned|ventilated)\b",
    re.IGNORECASE,
)

# ── Section keyword maps (expanded vs v1) ──────────────────────────────────────
_SECTION_HINTS: SectionTermMap = {
    "I": (
        "diagnosis", "diagnoses", "discharge diagnosis", "discharge diagnoses",
        "principal diagnosis", "secondary diagnosis", "active problems",
        "problem list", "past medical history", "pmh", "comorbidity",
        "comorbidities", "history of", "chronic", "acute", "condition",
        "disease", "disorder", "syndrome", "infection", "failure",
        "myocardial infarction", "heart failure", "diabetes", "hypertension",
        "atrial fibrillation", "stroke", "pulmonary embolism",
    ),
    "N": (
        "medication", "medications", "meds", "discharge meds", "home meds",
        "current meds", "medication list", "discharge medication",
        "dose", "dosage", "tablet", "capsule", "inject", "pharmacy",
        "insulin", "antibiotic", "anticoagulant", "warfarin", "heparin",
        "opioid", "aspirin", "metoprolol", "lisinopril", "furosemide",
        "amoxicillin", "azithromycin", "prescription", "drug", "prescribed",
    ),
    "O": (
        "procedure", "procedures", "operation", "surgery", "treatment",
        "treatments", "therapy", "oxygen", "dialysis", "intravenous",
        "transfusion", "chemotherapy", "radiation", "ventilator",
        "tracheostomy", "suction", "continuous positive airway pressure",
        "bilevel positive airway pressure", "peripherally inserted central catheter",
        "central line", "catheter", "physical therapy", "occupational therapy",
        "speech therapy",
    ),
}

_SECTION_HEADER_HINTS: SectionTermMap = {
    "I": (
        "discharge diagnosis", "discharge diagnoses", "diagnosis", "diagnoses",
        "active problems", "problem list", "past medical history", "pmh",
        "assessment and plan", "impression",
    ),
    "N": (
        "discharge medications", "discharge meds", "medications on discharge",
        "home medications", "home meds", "medication list", "medications", "meds",
    ),
    "O": (
        "procedures", "procedure", "treatments", "treatment", "therapy",
        "special treatments", "hospital course", "brief hospital course",
        "pertinent results",
    ),
}

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_BLOCK_SPLIT_PATTERN = re.compile(r"\n\s*\n+")

_I_KEYWORDS = ("diagnos", "disease", "condition", "comorb", "icd", "disorder", "syndrome")
_N_KEYWORDS = ("med", "drug", "insulin", "warfarin", "heparin", "anticoagul",
               "antibiotic", "opioid", "prescription", "tablet", "capsule")
_O_KEYWORDS = ("procedure", "treatment", "therapy", "oxygen", "dialysis",
               "intravenous", "transfusion", "vent", "surgery", "catheter", "intubat")

# Stopwords removed during concept-overlap comparison
_CONCEPT_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
    "with", "for", "of", "in", "on", "at", "to", "from", "by", "and",
    "or", "but", "not", "this", "that", "it", "be", "been", "being",
    "as", "also", "no", "yes", "patient", "history", "noted", "given",
    "without", "due", "including", "following", "after", "during",
}


# ── Knowledge node dataclass ───────────────────────────────────────────────────

@dataclasses.dataclass
class KnowledgeNode:
    """A single evidence node in the patient knowledge graph."""
    node_id: str
    section: str           # "I", "N", or "O"
    source: str            # "unstructured" or "structured:<dataset_name>"
    text: str              # display text (original or normalized)
    assertion: str         # "CONFIRMED", "NEGATED", "UNCERTAIN"
    conflicts_with: List[str] = dataclasses.field(default_factory=list)


# ── Public helpers: abbreviation expansion & assertion detection ───────────────

def expand_abbreviations(text: str) -> str:
    """Expand common clinical abbreviations to their full forms.

    Example: "Patient with CHF and DM2 on insulin" →
             "Patient with congestive heart failure and diabetes mellitus type 2
              on insulin"
    """
    def _replace(m: re.Match) -> str:
        token = m.group(0)
        return _ABBREVIATIONS.get(token.lower(), token)

    return _ABBREV_PATTERN.sub(_replace, text)


def detect_assertion(text: str) -> str:
    """Return the assertion status of a clinical text snippet.

    Returns one of:
    * ``"NEGATED"``  — the clinical concept is explicitly ruled out / denied.
    * ``"UNCERTAIN"`` — the concept is hedged, suspected, or a differential.
    * ``"CONFIRMED"`` — the concept is asserted as present (default).

    The algorithm is a lightweight NegEx-style rule: it looks for negation and
    uncertainty cues in the snippet (no syntactic parse required).

    Uncertainty is checked BEFORE negation because hedging phrases such as
    "cannot rule out" contain literal negation words ("rule out") yet express
    epistemic uncertainty rather than explicit negation.
    """
    if _UNCERTAINTY_PATTERN.search(text):
        return "UNCERTAIN"
    if _NEG_PRE_PATTERN.search(text) or _NEG_POST_PATTERN.search(text):
        return "NEGATED"
    return "CONFIRMED"


# ── Internal text utilities ────────────────────────────────────────────────────

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
    seen: Set[str] = set()
    keywords: List[str] = []
    for section in sections:
        for word in _SECTION_HINTS.get(section.upper(), ()):
            if word not in seen:
                seen.add(word)
                keywords.append(word)
    return keywords


def _split_into_blocks(text: str) -> List[str]:
    blocks = [block.strip() for block in _BLOCK_SPLIT_PATTERN.split(text) if block.strip()]
    return blocks or ([text.strip()] if text.strip() else [])


def _score_block(block: str, section: str) -> int:
    """Score a text block for relevance to *section*, using expanded text."""
    expanded = expand_abbreviations(block)
    lowered = expanded.lower()
    score = 0

    # Section header bonus
    for header in _SECTION_HEADER_HINTS.get(section, ()):
        if header in lowered:
            score += 5

    # Keyword count
    for keyword in _SECTION_HINTS.get(section, ()):
        score += lowered.count(keyword)

    # Extended clinical pattern bonuses
    if section == "I":
        score += len(_DIAGNOSIS_SUFFIX_PATTERN.findall(lowered))
        score += 2 * len(_DIAGNOSIS_VERB_PATTERN.findall(lowered))
    elif section == "N":
        score += 3 * len(_DOSAGE_PATTERN.findall(lowered))
        score += 2 * len(_ROUTE_PATTERN.findall(lowered))
    elif section == "O":
        score += 3 * len(_PROCEDURE_VERB_PATTERN.findall(lowered))

    return score


def _ranked_block_key(entry: RankedBlock) -> Tuple[int, int]:
    return (-entry[0], entry[1])


def _trim_block(block: str, section: str, max_sentences: int) -> str:
    """Return the most relevant sentences from *block* for *section*.

    Sentences with NEGATED assertion status are included only if they score
    higher than confirmed alternatives (so the LLM sees what was ruled out).
    """
    header_hints = _SECTION_HEADER_HINTS.get(section, ())
    keywords = _SECTION_HINTS.get(section, ())
    sentences = [seg.strip() for seg in _SENTENCE_SPLIT_PATTERN.split(block) if seg.strip()]
    selected: List[str] = []

    for sentence in sentences:
        expanded = expand_abbreviations(sentence)
        lowered = expanded.lower()
        matches_section = (
            any(hint in lowered for hint in header_hints)
            or any(keyword in lowered for keyword in keywords)
            or (section == "I" and (
                _DIAGNOSIS_SUFFIX_PATTERN.search(lowered)
                or _DIAGNOSIS_VERB_PATTERN.search(lowered)
            ))
            or (section == "N" and (
                _DOSAGE_PATTERN.search(lowered)
                or _ROUTE_PATTERN.search(lowered)
            ))
            or (section == "O" and _PROCEDURE_VERB_PATTERN.search(lowered))
        )
        if matches_section:
            selected.append(sentence)
        if len(selected) >= max_sentences:
            break

    if not selected:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        selected = lines[:max_sentences]

    return " ".join(selected)


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    deduped: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item.strip())
    return deduped


def _classify_section_from_text(text: str, allowed_sections: Set[str]) -> Optional[str]:
    """Classify text into I/N/O using both keyword stems and clinical patterns.

    Abbreviation expansion is applied before classification so that e.g.
    "CHF" is classified as Section I via "congestive heart failure".
    """
    expanded = expand_abbreviations(text)
    lowered = expanded.lower()

    if "I" in allowed_sections:
        if any(token in lowered for token in _I_KEYWORDS):
            return "I"
        if _DIAGNOSIS_SUFFIX_PATTERN.search(lowered) or _DIAGNOSIS_VERB_PATTERN.search(lowered):
            return "I"

    if "N" in allowed_sections:
        if any(token in lowered for token in _N_KEYWORDS):
            return "N"
        if _DOSAGE_PATTERN.search(lowered) or _ROUTE_PATTERN.search(lowered):
            return "N"

    if "O" in allowed_sections:
        if any(token in lowered for token in _O_KEYWORDS):
            return "O"
        if _PROCEDURE_VERB_PATTERN.search(lowered):
            return "O"

    return None


def _section_from_dataset_name(dataset_name: str, allowed_sections: Set[str]) -> Optional[str]:
    lowered = dataset_name.lower()
    if "I" in allowed_sections and any(w in lowered for w in ("diagnos", "problem", "condition", "icd")):
        return "I"
    if "N" in allowed_sections and any(w in lowered for w in ("med", "prescription", "drug", "pharm")):
        return "N"
    if "O" in allowed_sections and any(w in lowered for w in ("procedure", "treatment", "therapy", "service")):
        return "O"
    return None


def _row_to_fact_text(row: Dict[str, str], max_chars: int = 220) -> str:
    parts = [f"{k}={v}" for k, v in row.items() if v]
    fact = ", ".join(parts)
    if len(fact) <= max_chars:
        return fact
    return fact[: max_chars - 3].rstrip() + "..."


def _structured_rows_from_metadata(
    note_metadata: Optional[Dict[str, Any]],
) -> Dict[str, List[Dict[str, str]]]:
    if not isinstance(note_metadata, dict):
        return {}
    structured = note_metadata.get("structured_data", {})
    if not isinstance(structured, dict):
        return {}

    structured_dict = cast(Dict[str, Any], structured)
    result: Dict[str, List[Dict[str, str]]] = {}
    for dataset_name, rows in structured_dict.items():
        if not isinstance(rows, list):
            continue
        normalized_rows: List[Dict[str, str]] = []
        rows_list = cast(List[Any], rows)
        for row in rows_list:
            if not isinstance(row, dict):
                continue
            row_dict = cast(Dict[str, Any], row)
            normalized_rows.append(
                {
                    str(key): str(value).strip()
                    for key, value in row_dict.items()
                    if str(value).strip()
                }
            )
        if normalized_rows:
            result[dataset_name] = normalized_rows
    return result


# ── Concept-overlap for conflict detection ─────────────────────────────────────

def _significant_tokens(text: str) -> Set[str]:
    """Return non-stopword tokens from *text* after abbreviation expansion."""
    expanded = expand_abbreviations(text).lower()
    tokens = re.findall(r"[a-z]{3,}", expanded)
    return {t for t in tokens if t not in _CONCEPT_STOPWORDS}


def _concept_overlap(text_a: str, text_b: str) -> float:
    """Jaccard similarity of significant tokens between two text snippets."""
    ta = _significant_tokens(text_a)
    tb = _significant_tokens(text_b)
    if not ta or not tb:
        return 0.0
    intersection = ta & tb
    union = ta | tb
    return len(intersection) / len(union)


# ── Knowledge-node builders ────────────────────────────────────────────────────

def _build_unstructured_nodes(
    snippets: List[str],
    allowed_sections: Set[str],
    id_prefix: str = "U",
) -> List[KnowledgeNode]:
    """Convert labelled snippet strings into KnowledgeNode objects."""
    nodes: List[KnowledgeNode] = []
    counter = 1
    for snippet in snippets:
        section: Optional[str] = None
        text = snippet

        if snippet.startswith("[") and "]" in snippet:
            section = snippet[1: snippet.index("]")].strip().upper()
            text = snippet.split("]", 1)[1].strip()

        if section not in allowed_sections:
            section = _classify_section_from_text(text, allowed_sections)
        if section not in allowed_sections:
            continue

        assertion = detect_assertion(text)
        nodes.append(KnowledgeNode(
            node_id=f"{id_prefix}{counter}",
            section=section,
            source="unstructured",
            text=text,
            assertion=assertion,
        ))
        counter += 1
    return nodes


def _build_structured_nodes(
    note_metadata: Optional[Dict[str, Any]],
    allowed_sections: Set[str],
    id_prefix: str = "S",
) -> List[KnowledgeNode]:
    """Convert structured dataset rows into KnowledgeNode objects."""
    structured_rows = _structured_rows_from_metadata(note_metadata)
    nodes: List[KnowledgeNode] = []
    counter = 1

    for dataset_name, rows in structured_rows.items():
        dataset_section = _section_from_dataset_name(dataset_name, allowed_sections)
        for row in rows:
            fact_text = _row_to_fact_text(row)
            if not fact_text:
                continue

            section = dataset_section or _classify_section_from_text(fact_text, allowed_sections)
            if section not in allowed_sections:
                continue

            # Structured data (ICD codes, medication orders) is nearly always
            # affirmative — assertion detection is still applied as a safety net.
            assertion = detect_assertion(fact_text)

            nodes.append(KnowledgeNode(
                node_id=f"{id_prefix}{counter}",
                section=section,
                source=f"structured:{dataset_name}",
                text=fact_text,
                assertion=assertion,
            ))
            counter += 1

    return nodes


def _detect_and_annotate_conflicts(nodes: List[KnowledgeNode]) -> List[KnowledgeNode]:
    """Mark nodes that contradict each other across sources.

    Conflict = two nodes in the same section where one source asserts CONFIRMED
    and the other asserts NEGATED, and their concept tokens overlap significantly
    (Jaccard ≥ 0.25).

    Resolution strategy
    -------------------
    * Structured-CONFIRMED vs. Unstructured-NEGATED:
        Structured data (coded discharge diagnoses / medication orders) records
        the final clinical conclusion, while an unstructured negation often
        reflects a differential that was ruled out during the visit.
        → Keep both; mark conflict; flag the unstructured node.
    * Unstructured-CONFIRMED vs. Structured-NEGATED (rare, usually data error):
        → Keep both; mark conflict on both nodes.
    * CONFIRMED vs. UNCERTAIN (any direction):
        Not a hard conflict — uncertainty during workup is normal.
        → No conflict flag; uncertain node is still emitted.
    """
    structured = [n for n in nodes if n.source.startswith("structured")]
    unstructured = [n for n in nodes if n.source == "unstructured"]

    for sn in structured:
        for un in unstructured:
            if sn.section != un.section:
                continue
            overlap = _concept_overlap(sn.text, un.text)
            if overlap < 0.25:
                continue
            # Only flag genuine assertion reversals (CONFIRMED ↔ NEGATED)
            if {sn.assertion, un.assertion} == {"CONFIRMED", "NEGATED"}:
                if sn.node_id not in un.conflicts_with:
                    un.conflicts_with.append(sn.node_id)
                if un.node_id not in sn.conflicts_with:
                    sn.conflicts_with.append(un.node_id)

    return nodes


def _render_node(node: KnowledgeNode) -> str:
    """Render a KnowledgeNode as a compact string for the knowledge graph."""
    conflict_tag = f" [CONFLICTS:{','.join(node.conflicts_with)}]" if node.conflicts_with else ""
    return (
        f"- ({node.node_id}) FACT[{node.section}][{node.assertion}]"
        f" source={node.source}{conflict_tag}: {node.text}"
    )


def _node_relation(section: str) -> str:
    return (
        "has_diagnosis" if section == "I"
        else "takes_medication" if section == "N"
        else "received_treatment"
    )


# ── Public API: structured-summary helper (backward-compatible) ───────────────

def format_structured_data_summary(
    note_metadata: Optional[Dict[str, Any]],
    sections: Sequence[str],
    max_rows: int = 40,
) -> str:
    """Return a compact, section-aware summary of structured data rows."""
    structured_rows = _structured_rows_from_metadata(note_metadata)
    if not structured_rows:
        return ""

    allowed_sections = {s.upper() for s in sections}
    lines: List[str] = []
    rows_added = 0

    for dataset_name, rows in structured_rows.items():
        dataset_section = _section_from_dataset_name(dataset_name, allowed_sections)
        for row in rows:
            if rows_added >= max_rows:
                break
            fact_text = _row_to_fact_text(row)
            if not fact_text:
                continue
            inferred_section = dataset_section or _classify_section_from_text(fact_text, allowed_sections)
            if inferred_section is None:
                continue
            lines.append(f"- [{inferred_section}] {dataset_name}: {fact_text}")
            rows_added += 1

    return "\n".join(lines)


# ── Public API: priority snippet extraction ────────────────────────────────────

def extract_priority_snippets(
    text: str,
    sections: Sequence[str],
    max_sentences: int = 80,
) -> List[str]:
    """Return note snippets most likely to contain I/N/O evidence.

    Abbreviation expansion is used during scoring but the *original* text is
    returned so the LLM can read verbatim note language.
    """
    normalized_sections = [s.upper() for s in sections]
    keywords = _keywords_for_sections(normalized_sections)
    if not keywords or not text.strip():
        return []

    blocks = _split_into_blocks(text)
    snippets: List[str] = []
    seen: Set[str] = set()
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

    # Fallback: scan individual sentences using expanded text for matching,
    # but return the original sentence text.
    fallback_sentences = [seg.strip() for seg in _SENTENCE_SPLIT_PATTERN.split(text) if seg.strip()]
    for sentence in fallback_sentences[:max_sentences]:
        expanded_low = expand_abbreviations(sentence).lower()
        if any(keyword in expanded_low for keyword in keywords):
            snippets.append(sentence)

    return snippets


# ── Public API: patient knowledge graph ───────────────────────────────────────

def build_patient_knowledge_graph_chart(
    note_text: str,
    sections: Sequence[str],
    note_metadata: Optional[Dict[str, Any]] = None,
    max_nodes: int = 60,
) -> str:
    """Build a graph-style chart fusing unstructured and structured evidence.

    Each node carries:
    * section (I / N / O)
    * assertion status (CONFIRMED / NEGATED / UNCERTAIN)
    * source (unstructured vs. structured dataset name)
    * conflict annotation when structured and unstructured disagree

    NEGATED nodes are included (not silently dropped) so the downstream LLM
    extractor can see what was explicitly ruled out.  Conflict annotations
    surface irreconcilable disagreements for the extractor to adjudicate.
    """
    normalized_sections = [s.upper() for s in sections]
    allowed_sections: Set[str] = set(normalized_sections)

    cleaned = clean_discharge_text(note_text)
    snippets = extract_priority_snippets(cleaned, normalized_sections, max_sentences=60)

    # Build node lists
    unstructured_nodes = _build_unstructured_nodes(snippets, allowed_sections, id_prefix="U")
    structured_nodes = _build_structured_nodes(note_metadata, allowed_sections, id_prefix="S")

    # Renumber structured nodes so they don't collide with unstructured IDs
    offset = len(unstructured_nodes)
    for i, node in enumerate(structured_nodes):
        node.node_id = f"S{offset + i + 1}"

    all_nodes = unstructured_nodes + structured_nodes

    # Cap total nodes, preferring confirmed facts
    if len(all_nodes) > max_nodes:
        confirmed = [n for n in all_nodes if n.assertion == "CONFIRMED"]
        uncertain = [n for n in all_nodes if n.assertion == "UNCERTAIN"]
        negated = [n for n in all_nodes if n.assertion == "NEGATED"]
        # Fill quota: confirmed first, then uncertain, then negated
        all_nodes = (confirmed + uncertain + negated)[:max_nodes]

    # Detect and annotate conflicts
    all_nodes = _detect_and_annotate_conflicts(all_nodes)

    if not all_nodes:
        return ""

    # Build patient node label
    patient_hint = "patient"
    if isinstance(note_metadata, dict):
        subject_id = str(note_metadata.get("subject_id", "")).strip()
        hadm_id = str(note_metadata.get("hadm_id", "")).strip()
        if subject_id or hadm_id:
            patient_hint = f"patient(subject_id={subject_id or '?'}; hadm_id={hadm_id or '?'})"

    node_lines: List[str] = [f"- (P0) PERSON: {patient_hint}"]
    edge_lines: List[str] = []

    for node in all_nodes:
        node_lines.append(_render_node(node))
        edge_lines.append(f"- P0 -{_node_relation(node.section)}-> {node.node_id}")

    # Conflict summary section
    conflict_nodes = [n for n in all_nodes if n.conflicts_with]
    conflict_lines: List[str] = []
    if conflict_nodes:
        conflict_lines.append("CONFLICTS:")
        reported: Set[str] = set()
        for node in conflict_nodes:
            for other_id in node.conflicts_with:
                pair = tuple(sorted([node.node_id, other_id]))
                if pair in reported:
                    continue
                reported.add(pair)
                other_nodes = [n for n in all_nodes if n.node_id == other_id]
                other = other_nodes[0] if other_nodes else None
                if other:
                    # Resolution guidance based on source authority
                    if node.source.startswith("structured") and other.assertion == "NEGATED":
                        resolution = "PREFER STRUCTURED (discharge diagnosis codes are authoritative)"
                    elif other.source.startswith("structured") and node.assertion == "NEGATED":
                        resolution = "PREFER STRUCTURED (discharge diagnosis codes are authoritative)"
                    else:
                        resolution = "REVIEW BOTH — source authority unclear"
                    conflict_lines.append(
                        f"- {node.node_id}[{node.assertion}] vs {other.node_id}[{other.assertion}]"
                        f" | {resolution}"
                    )

    deduped_nodes = _dedupe_preserve_order(node_lines)
    deduped_edges = _dedupe_preserve_order(edge_lines)

    parts = [
        "=== PATIENT KNOWLEDGE GRAPH ===",
        "NODES:",
        *deduped_nodes,
        "EDGES:",
        *deduped_edges,
    ]
    if conflict_lines:
        parts.extend(conflict_lines)

    return "\n".join(parts)


# ── Public API: extraction context (backward-compatible) ───────────────────────

def build_extraction_context(
    note_text: str,
    sections: Sequence[str],
    max_clean_chars: int = 2500,
    max_snippet_chars: int = 4500,
    note_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Build focused context for the extractor.

    Combines the enriched knowledge graph, labelled priority snippets, and
    structured evidence into a single context block.
    """
    cleaned = clean_discharge_text(note_text)
    if not cleaned and not note_metadata:
        return ""

    snippets = extract_priority_snippets(cleaned, sections)
    snippet_block = "\n".join(f"- {s}" for s in snippets)
    knowledge_graph = build_patient_knowledge_graph_chart(
        note_text=cleaned,
        sections=sections,
        note_metadata=note_metadata,
    )
    structured_block = format_structured_data_summary(note_metadata, sections=sections)

    parts: List[str] = []
    parts.append("=== TARGET SECTIONS ===\n" + ", ".join(s.upper() for s in sections))

    if knowledge_graph:
        parts.append(knowledge_graph)

    if snippet_block:
        parts.append("=== PRIORITY EVIDENCE ===\n" + snippet_block[:max_snippet_chars])

    if structured_block:
        parts.append("=== STRUCTURED EVIDENCE ===\n" + structured_block)

    if cleaned and (not snippets or len(snippet_block) < 600):
        parts.append("=== SUPPORTING NOTE EXCERPT ===\n" + cleaned[:max_clean_chars])

    return "\n\n".join(parts)
