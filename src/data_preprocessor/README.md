# Data Preprocessor

This package loads MIMIC IV discharge notes and transforms raw clinical text into a structured **Patient Knowledge Graph** ready for extraction.

## Files

| File | Responsibility |
|------|---------------|
| `data_loader.py` | Load discharge notes from CSV/Excel; attach structured clinical tables by `hadm_id` |
| `preprocessor.py` | Clean text, expand abbreviations, detect assertion status, build knowledge graph, detect conflicts |

---

## data_loader.py

### Classes

#### `DischargeNote`

Represents a single discharge note record.

```python
from src.data_preprocessor.data_loader import DischargeNote

note = DischargeNote(
    note_id="1001",
    subject_id="10001",
    hadm_id="200001",
    text="78 yo male admitted with CHF exacerbation...",
    metadata={}
)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `note_id` | `str` | Unique identifier for the note record |
| `subject_id` | `str` | Patient identifier |
| `hadm_id` | `str` | Hospital admission identifier |
| `text` | `str` | Full free-text discharge note |
| `metadata` | `dict` | Optional — holds `structured_data` attached by the loader |

---

#### `MIMICDischargeLoader`

Loads discharge notes from a CSV or Excel file and optionally attaches structured clinical tables (diagnoses, prescriptions, procedures) keyed by `hadm_id`.

```python
from src.data_preprocessor.data_loader import MIMICDischargeLoader

loader = MIMICDischargeLoader(
    source="data/discharge.csv/discharge.csv",
    structured_sources={
        "diagnoses":     "data/diagnoses_icd.csv",
        "prescriptions": "data/prescriptions.csv",
        "procedures":    "data/procedures_icd.csv",
    },
)
notes = loader.load()   # → List[DischargeNote]
note = notes[0]
print(note.text[:200])
print(note.metadata["structured_data"]["diagnoses"])
```

**Constructor parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source` | required | Path to CSV/Excel discharge note file |
| `structured_sources` | `{}` | Dict mapping dataset name → CSV/Excel path |
| `note_id_col` | `"note_id"` | Column name for note ID |
| `subject_id_col` | `"subject_id"` | Column name for subject/patient ID |
| `hadm_id_col` | `"hadm_id"` | Column name for hospital admission ID |
| `text_col` | `"text"` | Column name for free-text note |
| `sample_size` | `None` | If set, load only the first N rows |
| `structured_join_priority` | `"hadm_id"` | Join key for attaching structured tables |

**Expected CSV/Excel column layout**

| `note_id` | `subject_id` | `hadm_id` | `text` |
|-----------|--------------|-----------|--------|
| 1001 | 10001 | 200001 | Patient is a 78-year-old male… |

---

## preprocessor.py

The preprocessor transforms raw clinical text into a structured context block through six sequential steps.

### Pipeline overview

```
Raw discharge text + structured tables
        │
 Step 1: Clean & normalize          remove placeholders, admin headers, excess whitespace
        │
 Step 2: Expand abbreviations       CHF → congestive heart failure, ASA → aspirin …
        │
 Step 3: Score evidence blocks      rank paragraphs by section (I / N / O) relevance
        │
 Step 4: Detect assertion status    CONFIRMED / NEGATED / UNCERTAIN (NegEx-style)
        │
 Step 5: Build knowledge graph      one node per evidence snippet or structured row
        │
 Step 6: Detect conflicts           Jaccard overlap between structured ↔ unstructured nodes
        │
        ▼
Patient Knowledge Graph string  →  passed to LLMExtractor or Seq2SeqExtractor
```

---

### Public API

```python
from src.data_preprocessor.preprocessor import (
    clean_discharge_text,
    expand_abbreviations,
    detect_assertion,
    extract_priority_snippets,
    format_structured_data_summary,
    build_patient_knowledge_graph_chart,
    build_extraction_context,
)
```

---

#### `clean_discharge_text(text) → str`

Removes MIMIC de-identification placeholders (`[**...**]`), administrative header lines (Name, DOB, Unit No, Admission/Discharge Date), repeated underscores, and excess whitespace.

```python
raw = "[**Patient Name**] admitted 01/01. Name: John Doe. Has hypertension."
clean_discharge_text(raw)
# → "Has hypertension."
```

---

#### `expand_abbreviations(text) → str`

Expands whole-word clinical abbreviations (case-insensitive) to their full forms. Used internally before scoring; original text is preserved for the extractor.

```python
expand_abbreviations("CHF, DM2, AFIB on warfarin PO BID.")
# → "congestive heart failure, diabetes mellitus type 2,
#    atrial fibrillation on warfarin oral by mouth twice daily."
```

Selected mappings (50+ total):

| Abbreviation | Expanded |
|---|---|
| `CHF` | congestive heart failure |
| `MI` / `STEMI` / `NSTEMI` | myocardial infarction (variants) |
| `AFIB` / `AF` | atrial fibrillation |
| `DM2` / `T2DM` | diabetes mellitus type 2 |
| `HTN` | hypertension |
| `ASA` / `APAP` | aspirin / acetaminophen |
| `PO` / `IV` / `SQ` | oral / intravenous / subcutaneous |
| `BID` / `TID` / `QD` | twice / three times / once daily |
| `CABG` / `PCI` | coronary artery bypass graft / percutaneous coronary intervention |
| `PICC` / `CVL` | peripherally inserted central catheter / central venous line |
| `CPAP` / `BiPAP` | continuous / bilevel positive airway pressure |
| `PT` / `OT` / `ST` | physical / occupational / speech therapy |

---

#### `detect_assertion(text) → "CONFIRMED" | "NEGATED" | "UNCERTAIN"`

NegEx-style assertion detection. Uncertainty is checked before negation so that phrases like *"cannot rule out sepsis"* are correctly classified as `UNCERTAIN` rather than `NEGATED`.

```python
detect_assertion("Patient has atrial fibrillation.")   # → "CONFIRMED"
detect_assertion("No evidence of PE.")                  # → "NEGATED"
detect_assertion("DVT was ruled out.")                  # → "NEGATED"
detect_assertion("Cannot rule out sepsis at this time.")# → "UNCERTAIN"
detect_assertion("Possible CHF exacerbation.")          # → "UNCERTAIN"
```

---

#### `extract_priority_snippets(text, sections, max_sentences) → List[str]`

Returns the most clinically relevant sentences for each requested MDS section, labelled with their section tag.

```python
snippets = extract_priority_snippets(note_text, sections=["I", "N", "O"])
# → ["[I] Discharge diagnosis: CHF exacerbation, AKI.",
#    "[N] Warfarin 5mg PO daily, furosemide 40mg BID.",
#    "[O] BiPAP initiated. PICC placed."]
```

---

#### `build_patient_knowledge_graph_chart(note_text, sections, note_metadata, max_nodes) → str`

Builds the full knowledge graph string with assertion tags, source labels, and conflict annotations.

```python
graph = build_patient_knowledge_graph_chart(
    note.text,
    sections=["I", "N", "O"],
    note_metadata=note.metadata,
)
print(graph)
```

**Example output**
```
=== PATIENT KNOWLEDGE GRAPH ===
NODES:
- (P0) PERSON: patient(subject_id=12345; hadm_id=99001)
- (U1) FACT[I][CONFIRMED] source=unstructured: Discharge diagnosis: CHF exacerbation.
- (U2) FACT[I][NEGATED]   source=unstructured: Patient denies any history of atrial fibrillation.
- (S3) FACT[I][CONFIRMED] source=structured:diagnoses: icd_code=I48.0 [CONFLICTS:U2]
EDGES:
- P0 -has_diagnosis-> U1
- P0 -has_diagnosis-> U2
- P0 -has_diagnosis-> S3
CONFLICTS:
- S3[CONFIRMED] vs U2[NEGATED] | PREFER STRUCTURED (discharge diagnosis codes are authoritative)
```

---

#### `build_extraction_context(note_text, sections, note_metadata, ...) → str`

Top-level function called by both extractors. Combines the knowledge graph, priority snippets, structured evidence summary, and a supporting note excerpt into a single context block.

```python
from src.data_preprocessor.preprocessor import build_extraction_context

context = build_extraction_context(
    note.text,
    sections=["I", "N", "O"],
    note_metadata=note.metadata,
)
print(context)
# → Full context block passed verbatim to LLMExtractor or Seq2SeqExtractor
```

---

### Conflict resolution

After all nodes are built, the preprocessor compares each structured node against each unstructured node in the same section using Jaccard similarity of significant tokens:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

If overlap ≥ 0.25 and assertion statuses are opposite (`CONFIRMED` ↔ `NEGATED`), both nodes are flagged and a resolution recommendation is embedded in the graph.

| Conflict type | Resolution |
|---|---|
| Structured `CONFIRMED` vs Unstructured `NEGATED` | Prefer structured — ICD discharge codes are authoritative |
| Structured `NEGATED` vs Unstructured `CONFIRMED` | Prefer structured (flag for review) |
| `CONFIRMED` vs `UNCERTAIN` (any direction) | Not a conflict — no flag emitted |

---

### Running tests

```bash
.conda/python.exe -m pytest tests/test_data_loader.py tests/test_preprocessor.py -v
```

Tests use no external API calls. The preprocessor tests cover:
- Administrative header and placeholder removal
- Abbreviation expansion and case insensitivity
- Assertion detection for confirmed, negated (pre/post cue), and uncertain patterns
- Knowledge graph conflict flagging
- Priority snippet extraction from abbreviation-only notes
