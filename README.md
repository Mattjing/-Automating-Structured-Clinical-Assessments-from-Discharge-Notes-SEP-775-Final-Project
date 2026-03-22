# Automating Structured Clinical Assessments from Discharge Notes

> **SEP-775 Final Project** — automatically extract and map free-text MIMIC IV discharge summaries to structured [MDS 3.0](https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/NursingHomeQualityInits/MDS30RAIManual) (Minimum Data Set) nursing-home assessment form fields using Large Language Models (LLMs).

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Data Preprocessing](#data-preprocessing)
   - [Inputs](#inputs)
   - [Pipeline](#preprocessing-pipeline)
   - [Challenges and Solutions](#challenges-and-solutions)
   - [Output: Patient Knowledge Graph](#output-patient-knowledge-graph)
   - [Public API](#public-api)
4. [Extraction and Mapping](#extraction-and-mapping)
5. [Supported MDS 3.0 Sections](#supported-mds-30-sections)
6. [Setup](#setup)
7. [GPU Support](#gpu-support)
8. [Usage](#usage)
9. [Configuration](#configuration)
10. [Running Tests](#running-tests)
11. [Data](#data)

---

## Overview

The system takes hospital discharge notes (unstructured free text) alongside structured clinical tables (diagnoses, medications, procedures) and automatically fills the MDS 3.0 form fields for **Section I (Active Disease Diagnoses)**, **Section N (Medications)**, and **Section O (Special Treatments, Procedures, Programs)**.

```
┌─────────────────────────────────────────────────────┐
│                     INPUT DATA                       │
│                                                      │
│  Unstructured                  Structured            │
│  ┌──────────────────┐    ┌──────────────────────┐   │
│  │ Discharge notes  │    │ diagnoses.csv         │   │
│  │ (free text)      │    │ prescriptions.csv     │   │
│  └────────┬─────────┘    │ procedures.csv  …     │   │
│           │              └──────────┬───────────┘   │
└───────────┼─────────────────────────┼───────────────┘
            │                         │
            ▼                         ▼
    MIMICDischargeLoader  ←── attaches structured rows
            │                   to each note via hadm_id
            ▼
    ┌───────────────────────────────────────────────┐
    │            DATA PREPROCESSOR                  │
    │  1. Clean & normalize text                    │
    │  2. Expand clinical abbreviations             │
    │  3. Detect assertion status (NegEx)           │
    │  4. Score & extract I/N/O evidence blocks     │
    │  5. Build patient knowledge graph             │
    │  6. Detect structured ↔ unstructured conflicts│
    └──────────────┬────────────────────────────────┘
                   │
                   ▼
         Patient Knowledge Graph
         (nodes tagged with section,
          assertion, source, conflicts)
                   │
          ┌────────┴────────┐
          ▼                 ▼
    LLMExtractor     MedBERTExtractor
    (GPT / OpenAI)   (biomedical NER)
          │                 │
          └────────┬────────┘
                   ▼
         MDSMapper / MedBERTMapper
                   │
                   ▼
           MDSAssessment
       (JSON / CSV / Excel)
```

---

## Project Structure

```
.
├── config/
│   ├── config.yaml                   # Runtime configuration
│   └── medbert_patterns.auto.json    # Auto-generated MedBERT regex seed patterns
├── scripts/
│   ├── extract_mds_sections_from_pdf.py   # Parse MDS 3.0 PDF → item text + patterns
│   └── preview_input_data.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Load discharge notes + attach structured tables
│   ├── mds_schema.py         # MDS 3.0 form field definitions
│   ├── preprocessor.py       # Core preprocessing: clean, expand, classify, graph
│   ├── extractor.py          # LLM-based field extraction (OpenAI)
│   ├── medbert_extractor.py  # MedBERT biomedical NER extraction
│   ├── mapper.py             # Validate and map LLM outputs to MDS schema
│   ├── medbert_mapper.py     # Validate/map MedBERT outputs to MDS schema
│   └── pipeline.py           # End-to-end orchestration
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_extractor.py
│   ├── test_mapper.py
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

---

## Data Preprocessing

Preprocessing is the most critical step in the pipeline. Raw clinical notes are noisy, abbreviated, and contain statements that are explicitly negated or merely suspected. Without careful preprocessing, the extractor will see irrelevant text, miss important facts written as abbreviations, and may treat ruled-out conditions as confirmed diagnoses.

All preprocessing logic lives in [src/preprocessor.py](src/preprocessor.py).

### Inputs

The preprocessor receives **two complementary types of data** for each patient admission:

#### 1. Unstructured input — discharge note (free text)

A hospital discharge note is a long, free-form clinical narrative written by the attending physician. It typically contains:

| Section | Example content |
|---------|----------------|
| Chief complaint / HPI | "78 yo male admitted with acute SOB and CP, hx of CHF." |
| Past medical history | "PMH: DM2, HTN, CAD s/p CABG 2019, AFIB on warfarin." |
| Hospital course | "Patient was intubated, placed on BiPAP, started on IV furosemide." |
| Discharge diagnosis | "1. CHF exacerbation  2. AKI  3. Atrial fibrillation" |
| Discharge medications | "Warfarin 5mg PO QD, Furosemide 40mg PO BID, ASA 81mg PO QD" |
| Special treatments | "Received 2 units PRBCs, dialysis initiated, PICC placed." |

This text is passed in as a raw Python string.

#### 2. Structured input — clinical data tables

Alongside the note, structured tabular datasets (CSV/Excel) can be attached by the data loader using the hospital admission ID (`hadm_id`) as a join key. These tables may include:

| Dataset name | Typical columns | Maps to section |
|--------------|-----------------|-----------------|
| `diagnoses`  | `icd_code`, `icd_version`, `description` | I |
| `prescriptions` | `drug`, `dose`, `route`, `frequency` | N |
| `procedures` | `icd_code`, `procedure_description` | O |
| `labevents`  | `itemid`, `label`, `value`, `unit` | O (some I) |

These tables arrive as Python dictionaries in `note.metadata["structured_data"]`.

**Example of what the loader attaches:**

```python
note.metadata = {
    "subject_id": "12345",
    "hadm_id": "99001",
    "structured_data": {
        "diagnoses": [
            {"icd_code": "I50.9", "description": "Heart failure, unspecified"},
            {"icd_code": "I48.0", "description": "Paroxysmal atrial fibrillation"},
        ],
        "prescriptions": [
            {"drug": "Warfarin",   "dose": "5 mg",  "route": "PO", "frequency": "Daily"},
            {"drug": "Furosemide", "dose": "40 mg", "route": "PO", "frequency": "BID"},
        ],
    }
}
```

---

### Preprocessing Pipeline

The preprocessor transforms raw inputs into a **Patient Knowledge Graph** through six sequential steps:

```
Raw discharge text + structured tables
            │
     Step 1: Clean & normalize
            │   • Remove de-identification placeholders ([**...**])
            │   • Strip administrative headers (Name, DOB, Unit No)
            │   • Collapse excess whitespace and newlines
            │
     Step 2: Expand clinical abbreviations
            │   • "CHF" → "congestive heart failure"
            │   • "MI"  → "myocardial infarction"
            │   • "ASA 81mg PO QD" → "aspirin 81mg oral by mouth once daily"
            │   (expanded text used for classification; original text preserved)
            │
     Step 3: Score and extract evidence blocks (per section I / N / O)
            │   • Split note into paragraph blocks
            │   • Score each block: section headers (+5), keyword hits (+1 each),
            │     medical suffix patterns, dosage patterns, procedure verbs
            │   • Select top-3 blocks per section
            │   • Trim each block to its most relevant sentences
            │
     Step 4: Detect assertion status (NegEx)
            │   • CONFIRMED — no negation or uncertainty cues
            │   • NEGATED   — "no evidence of", "ruled out", "patient denies"
            │   • UNCERTAIN — "possible", "suspected", "cannot rule out"
            │   (uncertainty checked before negation to handle "cannot rule out")
            │
     Step 5: Build knowledge graph nodes
            │   • One node per evidence snippet (unstructured) or data row (structured)
            │   • Each node carries: section, source, text, assertion status
            │
     Step 6: Detect and annotate conflicts
            │   • Compute token-level Jaccard overlap between structured and
            │     unstructured nodes in the same section
            │   • If overlap ≥ 0.25 AND assertions are opposite (CONFIRMED ↔ NEGATED)
            │     → mark both nodes with conflict references
            │   • Apply resolution heuristic (see Conflict Resolution section below)
            │
            ▼
    Patient Knowledge Graph (string)
    → passed to LLMExtractor or MedBERTExtractor
```

---

### Challenges and Solutions

Clinical text presents four major challenges that a naive keyword-matching approach cannot handle. Each is addressed by a dedicated mechanism in the preprocessor.

---

#### Challenge 1: Clinical Abbreviations

**Problem.** Clinicians write almost exclusively in abbreviations. A note may say:
> "PMH: CHF, DM2, AFIB. On warfarin, furosemide, ASA. Underwent CABG 2019. PICC placed. BiPAP initiated."

A keyword search for "congestive heart failure" would find nothing; a search for "medication" would find nothing. The note is clinically rich but lexically invisible to simple matchers.

**Solution: `expand_abbreviations(text)`**

Before any classification or scoring, all text is passed through an abbreviation expander that replaces ~50 common clinical abbreviations with their full forms using whole-word regex matching (so "AFIB" becomes "atrial fibrillation" but "afibrinogenemia" is left untouched).

```python
from src.preprocessor import expand_abbreviations

expand_abbreviations("PMH: CHF, DM2, AFIB. ASA 81mg PO QD.")
# → "past medical history: congestive heart failure, diabetes mellitus type 2,
#    atrial fibrillation. aspirin 81mg oral by mouth once daily."
```

The expanded text is used only for **classification and scoring**. The original note text is preserved and passed to the extractor so the LLM reads verbatim clinical language.

**Selected abbreviation mappings (50+ total):**

| Abbreviation | Expanded form | Section |
|---|---|---|
| `CHF` | congestive heart failure | I |
| `MI` / `STEMI` / `NSTEMI` | myocardial infarction (variants) | I |
| `COPD` | chronic obstructive pulmonary disease | I |
| `AFIB` / `AF` | atrial fibrillation | I |
| `DM2` / `T2DM` | diabetes mellitus type 2 | I |
| `HTN` | hypertension | I |
| `CKD` / `ESRD` | chronic kidney disease / end-stage renal disease | I |
| `DVT` / `PE` | deep vein thrombosis / pulmonary embolism | I |
| `CVA` / `TIA` | cerebrovascular accident stroke / transient ischemic attack | I |
| `ASA` / `APAP` | aspirin / acetaminophen | N |
| `ABX` | antibiotics | N |
| `PO` / `IV` / `SQ` | oral by mouth / intravenous / subcutaneous | N |
| `BID` / `TID` / `QD` | twice daily / three times daily / once daily | N |
| `CABG` / `PCI` | coronary artery bypass graft / percutaneous coronary intervention | O |
| `PICC` / `CVL` | peripherally inserted central catheter / central venous line | O |
| `CPAP` / `BiPAP` | continuous / bilevel positive airway pressure | O |
| `VENT` | mechanical ventilator | O |
| `PT` / `OT` / `ST` | physical / occupational / speech therapy | O |

---

#### Challenge 2: Synonyms and Clinical Paraphrases

**Problem.** The same clinical concept appears in many forms. "Atrial fibrillation" may be written as "AF", "AFib", "a-fib", "irregular rhythm", "paroxysmal AF", or coded as ICD I48. No fixed keyword list covers all variants.

**Solution: Extended pattern matching**

In addition to keyword lists, three classes of **clinical patterns** are applied to the abbreviation-expanded text:

| Pattern class | What it catches | Example |
|---|---|---|
| **Medical suffix patterns** | Condition names with diagnostic suffixes | `hepatitis`, `cardiomyopathy`, `thrombocytemia`, `nephrolithiasis` |
| **Dosage / route patterns** | Medication sentences without "medication" keyword | `"5 mg PO"`, `"40 mcg IV"`, `"2 units subcutaneous"` |
| **Procedure verb patterns** | Treatment sentences without "procedure" keyword | `"was intubated"`, `"underwent dialysis"`, `"PICC placed"`, `"transfused 2u PRBCs"` |

```python
# Medical suffix pattern catches condition names automatically
# even if "diagnosis" never appears in the sentence
"Patient presented with acute hepatitis and thrombocytemia."
#                           ↑ -itis           ↑ -emia  → Section I

# Dosage pattern catches medication sentences without headers
"Furosemide 40 mg IV twice daily was continued."
#           ↑↑↑↑↑↑↑↑↑ dosage match             → Section N

# Procedure verb catches treatment sentences
"Patient underwent emergent dialysis on hospital day 2."
#           ↑↑↑↑↑↑↑↑↑ verb match                  → Section O
```

---

#### Challenge 3: Negation and Uncertainty

**Problem.** Clinical notes mention conditions in three distinct epistemic modes:
- **Confirmed:** "Patient has atrial fibrillation."
- **Negated:** "DVT was ruled out on lower extremity Doppler."
- **Uncertain:** "Possible CHF exacerbation; cannot rule out PE."

A naive extractor that ignores this distinction will incorrectly report DVT and PE as confirmed diagnoses, inflating Section I and producing clinically dangerous errors.

**Solution: `detect_assertion(text)` — NegEx-style assertion detection**

Every evidence snippet is tagged with an assertion status before being added to the knowledge graph. The function uses two compiled regular-expression sets:

**Negation cue patterns (pre-concept):**
```
no · not · without · denies · no evidence of · no signs of ·
no history of · ruled out · negative for · absent · absence of ·
never · none · free of · not consistent with · unremarkable for
```

**Negation cue patterns (post-concept):**
```
ruled out · not present · not identified · not detected ·
not found · not seen · not noted · negative · not confirmed
```

**Uncertainty cue patterns:**
```
possible · possibly · probable · probably · likely ·
suspected · suspicion of · may have · might have ·
questionable · question of · concerning for · cannot rule out ·
could represent · appears to be · thought to be ·
presumed · differential includes · r/o · versus
```

**Key design decision: uncertainty is checked before negation.**

Hedging phrases like *"cannot rule out sepsis"* contain the literal word *"rule out"* (a negation cue), yet they express epistemic uncertainty — the clinician is saying the condition *cannot be excluded*. Checking uncertainty first ensures these are correctly classified as `UNCERTAIN` rather than `NEGATED`.

```python
from src.preprocessor import detect_assertion

detect_assertion("Patient has atrial fibrillation.")
# → "CONFIRMED"

detect_assertion("No evidence of pulmonary embolism on CT.")
# → "NEGATED"

detect_assertion("DVT was ruled out.")
# → "NEGATED"

detect_assertion("Cannot rule out sepsis at this time.")
# → "UNCERTAIN"  ← uncertainty checked first; "rule out" alone would give NEGATED

detect_assertion("Possible CHF exacerbation.")
# → "UNCERTAIN"
```

**How assertion status affects the knowledge graph:**

`NEGATED` nodes are **not silently dropped**. They are kept in the graph and passed to the extractor so the LLM can see what was explicitly ruled out. This prevents false positives when a differential diagnosis was considered and dismissed during the admission.

---

#### Challenge 4: Structured vs. Unstructured Conflicts

**Problem.** Structured data and free-text notes sometimes contradict each other. For example:

- Structured `diagnoses` table contains ICD I48 (atrial fibrillation) — a confirmed discharge diagnosis.
- The discharge note contains: *"Patient denies any prior history of atrial fibrillation."*

These appear to disagree, but they are capturing different things: the structured code records the *final discharge diagnosis*, while the note may be documenting the *patient's reported history* at admission. Without resolving this conflict, the extractor receives contradictory signals and may produce unreliable output.

**Solution: Token-overlap conflict detection with source-authority resolution**

After all nodes are built, the preprocessor compares **each structured node against each unstructured node in the same section** using Jaccard similarity of significant (non-stopword) tokens on abbreviation-expanded text:

```
Jaccard(tokens_A, tokens_B) = |A ∩ B| / |A ∪ B|
```

If the overlap exceeds **0.25** and the assertion statuses are opposite (`CONFIRMED` ↔ `NEGATED`), both nodes are marked with `conflicts_with` references and a resolution recommendation is added to the graph output.

**Resolution heuristic:**

| Conflict type | Resolution |
|---|---|
| Structured `CONFIRMED` vs. Unstructured `NEGATED` | **Prefer structured** — ICD discharge diagnosis codes are the authoritative final record; unstructured negation often reflects a ruled-out differential during the workup |
| Structured `NEGATED` vs. Unstructured `CONFIRMED` | **Prefer structured** (rare; may indicate documentation error — flag for review) |
| `CONFIRMED` vs. `UNCERTAIN` (any direction) | Not a conflict — uncertainty during workup is clinically normal; no flag emitted |

The resolution guidance is embedded in the graph as a human-readable annotation visible to the downstream LLM extractor, allowing it to make an informed decision.

---

### Output: Patient Knowledge Graph

The preprocessor emits a structured text block — the **Patient Knowledge Graph** — that becomes the primary input context for the extractor. Each node explicitly states:
- Which MDS section it belongs to (`I`, `N`, or `O`)
- Its assertion status (`CONFIRMED`, `NEGATED`, or `UNCERTAIN`)
- Its source (`unstructured` or `structured:<dataset_name>`)
- Any conflict references

**Example output for a patient with CHF, warfarin, and a noted conflict:**

```
=== TARGET SECTIONS ===
I, N, O

=== PATIENT KNOWLEDGE GRAPH ===
NODES:
- (P0) PERSON: patient(subject_id=12345; hadm_id=99001)
- (U1) FACT[I][CONFIRMED] source=unstructured: Discharge diagnosis: CHF exacerbation. AKI.
- (U2) FACT[I][NEGATED] source=unstructured: Patient denies any history of atrial fibrillation.
- (U3) FACT[N][CONFIRMED] source=unstructured: Warfarin 5mg PO daily, furosemide 40mg PO BID.
- (U4) FACT[O][CONFIRMED] source=unstructured: Patient received 2 units PRBCs. PICC placed.
- (S5) FACT[I][CONFIRMED] source=structured:diagnoses: icd_code=I48.0, description=atrial fibrillation [CONFLICTS:U2]
- (S6) FACT[N][CONFIRMED] source=structured:prescriptions: drug=Warfarin, dose=5 mg, route=PO
EDGES:
- P0 -has_diagnosis-> U1
- P0 -has_diagnosis-> U2
- P0 -takes_medication-> U3
- P0 -received_treatment-> U4
- P0 -has_diagnosis-> S5
- P0 -takes_medication-> S6
CONFLICTS:
- S5[CONFIRMED] vs U2[NEGATED] | PREFER STRUCTURED (discharge diagnosis codes are authoritative)

=== PRIORITY EVIDENCE ===
- [I] Discharge diagnosis: CHF exacerbation. AKI.
- [N] Warfarin 5mg PO daily, furosemide 40mg PO BID.
- [O] Patient received 2 units PRBCs. PICC placed.

=== STRUCTURED EVIDENCE ===
- [I] diagnoses: icd_code=I48.0, description=atrial fibrillation
- [N] prescriptions: drug=Warfarin, dose=5 mg, route=PO, frequency=Daily
```

This context block is then sent verbatim to the LLM extractor as the "discharge note" content.

---

### Public API

All preprocessing functions can be used independently:

```python
from src.preprocessor import (
    clean_discharge_text,       # Remove noise, placeholders, admin headers
    expand_abbreviations,       # Expand clinical abbreviations in text
    detect_assertion,           # NegEx: CONFIRMED / NEGATED / UNCERTAIN
    extract_priority_snippets,  # Top I/N/O evidence sentences from a note
    format_structured_data_summary,       # Compact structured-data text block
    build_patient_knowledge_graph_chart,  # Full knowledge graph string
    build_extraction_context,   # Full context block for the extractor
)
```

**`clean_discharge_text(text) → str`**

Removes de-identification placeholders (`[**...**]`), administrative header lines (Name, DOB, Unit No, Admission/Discharge Date), repeated underscores, and excess whitespace.

```python
raw = "[**Patient Name**] admitted 01/01. Name: John Doe. Has hypertension."
clean_discharge_text(raw)
# → "Has hypertension."
```

**`expand_abbreviations(text) → str`**

Expands whole-word clinical abbreviations (case-insensitive) to their full forms.

```python
expand_abbreviations("CHF, DM2, AFIB on warfarin PO BID.")
# → "congestive heart failure, diabetes mellitus type 2,
#    atrial fibrillation on warfarin oral by mouth twice daily."
```

**`detect_assertion(text) → "CONFIRMED" | "NEGATED" | "UNCERTAIN"`**

Returns the assertion status of a clinical sentence using NegEx-style rules.

```python
detect_assertion("No evidence of PE.")        # → "NEGATED"
detect_assertion("Possible CHF.")             # → "UNCERTAIN"
detect_assertion("Diagnosed with COPD.")      # → "CONFIRMED"
```

**`extract_priority_snippets(text, sections, max_sentences) → List[str]`**

Returns labelled sentence snippets most relevant to the requested MDS sections. Abbreviation expansion is used during scoring; original text is returned.

```python
snippets = extract_priority_snippets(note_text, sections=["I", "N", "O"])
# → ["[I] Discharge diagnosis: CHF exacerbation, AKI.",
#    "[N] Warfarin 5mg PO daily, furosemide 40mg BID.",
#    "[O] BiPAP initiated. PICC placed."]
```

**`build_patient_knowledge_graph_chart(note_text, sections, note_metadata, max_nodes) → str`**

Builds the full knowledge graph string combining unstructured evidence nodes and structured data nodes, with assertion tags and conflict annotations.

**`build_extraction_context(note_text, sections, note_metadata, ...) → str`**

Top-level function called by the extractor. Returns the complete context block (knowledge graph + priority snippets + structured evidence + supporting note excerpt).

---

## Extraction and Mapping

After preprocessing, the context is passed to one of two extractors:

### LLMExtractor (default — OpenAI GPT)

Sends the knowledge graph context to an OpenAI Chat Completions model with a structured system prompt that instructs it to output valid JSON mapping MDS item IDs to coded values.

Two preprocessing modes are supported and can be compared:
- **`heuristic`** — uses `build_extraction_context()` (the preprocessor above)
- **`llm_evidence`** — uses the LLM itself for a lightweight first-pass condensation before final extraction

### MedBERTExtractor (biomedical NER)

Uses a Hugging Face token-classification model (default: `d4data/biomedical-ner-all`) to identify named entities (diseases, drugs, procedures) in the preprocessed note. Entity spans are then mapped to MDS item IDs via regex patterns auto-generated from the MDS 3.0 PDF.

---

## Supported MDS 3.0 Sections

| Section | Title |
|---------|-------|
| A | Identification Information |
| B | Hearing, Speech, and Vision |
| C | Cognitive Patterns |
| D | Mood (PHQ-9) |
| E | Behavior |
| G | Functional Status (ADLs) |
| H | Bladder and Bowel |
| **I** | **Active Disease Diagnoses** |
| J | Health Conditions (Pain, Dyspnea, Falls) |
| K | Swallowing / Nutritional Status |
| M | Skin Conditions / Pressure Ulcers |
| **N** | **Medications** |
| **O** | **Special Treatments, Procedures, Programs** |

Sections I, N, and O are the primary focus of this pipeline.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

---

## GPU Support

GPU acceleration is used by the **MedBERT extractor** to speed up biomedical NER inference. The pipeline automatically detects CUDA and falls back to CPU if no GPU is found — no code changes are needed.

### Checking GPU readiness

Run the bundled diagnostic script before starting any training or inference workload:

```bash
python scripts/check_gpu.py
```

#### GPU available — expected output

```
torch_version: 2.x.x+cu118
torch_cuda_build: 11.8
cuda_available: True
cuda_device_count: 1
cuda_device_0: NVIDIA GeForce RTX 3080 (compute_capability=8.6)
gpu_tensor_op: PASS
```

Key indicators of a healthy GPU environment:

| Field | Expected value |
|---|---|
| `cuda_available` | `True` |
| `cuda_device_count` | ≥ 1 |
| `cuda_device_0` | Your GPU name and compute capability |
| `gpu_tensor_op` | `PASS` — a real 1024×1024 matrix multiply executed on CUDA |

#### GPU not available — expected output

```
torch_version: 2.x.x+cpu
torch_cuda_build: None
cuda_available: False
No CUDA runtime/device detected. Workloads will run on CPU.
```

The script exits with:
- `0` — GPU detected and tensor op passed
- `1` — `torch` could not be imported
- `2` — CUDA not available (CPU-only install or no compatible GPU)
- `3` — CUDA available but tensor operation failed

If CUDA is not detected, install a CUDA-enabled PyTorch build matching your driver version from [pytorch.org](https://pytorch.org/get-started/locally/).

---

### MedBERT device selection at runtime

When `MedBERTExtractor` (or the pipeline) initialises the NER model, it logs which device it selected. You will see one of the following messages:

| Scenario | Log message |
|---|---|
| GPU found and selected | `MedBERT using GPU (CUDA device 0: NVIDIA GeForce RTX 3080).` |
| Requested device index out of range — fell back to device 0 | `Requested CUDA device 2 is unavailable (device_count=1). Falling back to 0.` then `MedBERT using GPU (CUDA device 0: …).` |
| GPU preference explicitly disabled (`--medbert-force-cpu`) | `MedBERT using CPU (GPU preference disabled).` |
| CUDA not available at runtime | `MedBERT using CPU (CUDA not available).` |
| No CUDA devices detected | `MedBERT using CPU (no CUDA devices detected).` |
| `torch` import failed | `MedBERT using CPU (torch import failed: …).` |
| CUDA check raised an exception | `MedBERT using CPU (CUDA check failed: …).` |

Logs are emitted at `INFO` level. To see them, run with the default logging configuration or set `--log-level INFO`.

---

### Controlling GPU usage

**Python API**

```python
from src.medbert_extractor import MedBERTExtractor
from src.mds_schema import MDSSchema

schema = MDSSchema(section_ids=["I", "N", "O"])

# Use GPU (default) — picks CUDA device 0 when available
extractor = MedBERTExtractor(schema=schema, prefer_gpu=True, gpu_device=0)

# Explicit GPU device (e.g. second GPU in a multi-GPU machine)
extractor = MedBERTExtractor(schema=schema, prefer_gpu=True, gpu_device=1)

# Force CPU regardless of what hardware is present
extractor = MedBERTExtractor(schema=schema, prefer_gpu=False)
```

Via `ExtractionPipeline`:

```python
from src.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline(
    source="data/discharge.csv/discharge.csv",
    sections=["I", "N", "O"],
    medbert_prefer_gpu=True,   # default: True
    medbert_gpu_device=0,      # default: 0
)
assessments = pipeline.run()
```

**CLI flags**

```bash
# Explicitly enable GPU (default behaviour)
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode medbert-full --medbert-prefer-gpu

# Force CPU even if a GPU is present
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode medbert-full --medbert-force-cpu

# Target a specific CUDA device index
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode medbert-full --medbert-gpu-device 1
```

**`config/config.yaml`**

```yaml
extraction:
  medbert_prefer_gpu: true   # set to false to always use CPU
  medbert_gpu_device: 0      # CUDA device index
```

CLI flags take priority over config values when both are provided.

---

## Usage

### Quick start — from a CSV file

```python
from src.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline(
    source="data/discharge.csv/discharge.csv",
    openai_api_key="sk-...",          # or set OPENAI_API_KEY env var
    sections=["I", "N", "O"],
    output_dir="output",
    output_format="json",
    sample_size=5,                     # process first 5 notes by default
)
assessments = pipeline.run()
```

### Attaching structured data tables

```python
pipeline = ExtractionPipeline(
    source="data/discharge.csv/discharge.csv",
    structured_sources={
        "diagnoses":     "data/diagnoses_icd.csv",
        "prescriptions": "data/prescriptions.csv",
        "procedures":    "data/procedures_icd.csv",
    },
    openai_api_key="sk-...",
    sections=["I", "N", "O"],
    output_dir="output",
)
assessments = pipeline.run()
```

Structured tables are joined to each note via `hadm_id` (configurable with `structured_join_priority`).

### Using the preprocessor directly

```python
from src.data_loader import MIMICDischargeLoader
from src.preprocessor import build_extraction_context, detect_assertion, expand_abbreviations

loader = MIMICDischargeLoader(
    source="data/discharge.csv/discharge.csv",
    structured_sources={"diagnoses": "data/diagnoses_icd.csv"},
)
notes = loader.load()
note = notes[0]

# Inspect abbreviation expansion
expanded = expand_abbreviations(note.text[:500])

# Check assertion on a sentence
status = detect_assertion("No evidence of pulmonary embolism.")
# → "NEGATED"

# Build the full context that will be fed to the extractor
context = build_extraction_context(
    note.text,
    sections=["I", "N", "O"],
    note_metadata=note.metadata,
)
print(context)
```

### MedBERT extractor (no API key required)

```python
from src.data_loader import MIMICDischargeLoader
from src.mds_schema import MDSSchema
from src.medbert_extractor import MedBERTExtractor
from src.medbert_mapper import MedBERTMapper

loader = MIMICDischargeLoader(source="data/discharge.csv/discharge.csv")
notes = loader.load()

schema = MDSSchema(section_ids=["I", "N", "O"])
extractor = MedBERTExtractor(
    schema=schema,
    model_name="d4data/biomedical-ner-all",
    sections=["I", "N", "O"],
)
mapper = MedBERTMapper(schema=schema)

note = notes[0]
raw = extractor.extract(note.text, note_metadata=note.metadata)
assessment = mapper.map(note.note_id, note.subject_id, note.hadm_id, raw)
print(assessment.to_dict())
```

### Input CSV/Excel format

The discharge note file must contain at least these columns (all names are configurable):

| note_id | subject_id | hadm_id | text |
|---------|------------|---------|------|
| 1001 | 10001 | 200001 | Patient is a 78-year-old male … |

### CLI — interactive mode

```bash
python src/pipeline.py --source data/discharge.csv/discharge.csv
```

```
Select processing mode:
  --- LLM (OpenAI GPT) ---
  1. sample          - run the initial sample only
  2. sample-compare  - run the sample and compare preprocessing methods
  3. full            - process the entire dataset
  4. full-compare    - process the entire dataset and compare preprocessing methods
  --- MedBERT (biomedical NER, no API key required) ---
  5. medbert-sample  - run the initial sample with MedBERT NER
  6. medbert-full    - process the entire dataset with MedBERT NER
Enter mode [1-6] (default 1):
```

| Mode | Notes processed | Comparison output |
|------|-----------------|-------------------|
| `sample` | First `--sample-size` notes (default 5) | No |
| `sample-compare` | First `--sample-size` notes | Yes — diff CSV + JSON |
| `full` | All notes | No |
| `full-compare` | All notes | Yes — diff CSV + JSON |
| `medbert-sample` | First `--sample-size` notes | No (MedBERT backend) |
| `medbert-full` | All notes | No (MedBERT backend) |

### CLI — scripted / non-interactive use

```bash
# Sample run with LLM
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode sample

# Full dataset with preprocessing comparison
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode full-compare

# Full dataset using MedBERT (no API key needed)
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode medbert-full

# Force MedBERT onto CUDA device 1
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode medbert-full --medbert-gpu-device 1

# Force CPU (debug/fallback)
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode medbert-full --medbert-force-cpu

# Custom sample size
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode sample --sample-size 20
```

---

## Configuration

Edit `config/config.yaml` to adjust settings without touching source code.

```yaml
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  temperature: 0.0
  max_tokens: 2048

data:
  csv_path: "data/discharge.csv/discharge.csv"

preprocess:
  enabled: true

extraction:
    medbert_prefer_gpu: true
    medbert_gpu_device: 0

output:
  output_dir: "output"
  format: "json"
```

Output files written when `format: json`:

| File | Contents |
|------|----------|
| `output/mds_assessments.json` | Full MDS assessment objects |
| `output/mds_ino_form_ready.json` | Compact I/N/O codes + confidence scores |
| `output/mds_ino_preprocessing_comparison.json` | Side-by-side heuristic vs. LLM-evidence results |
| `output/mds_ino_preprocessing_diff_summary.json` | Only the fields that differ between modes |
| `output/mds_ino_preprocessing_diff_summary.csv` | Same, flattened for easy review |

---

## Running Tests

```bash
# Using the conda environment (recommended)
.conda/python.exe -m pytest tests/ -v

# Or with system Python if dependencies are installed
pytest tests/ -v
```

All tests use mocks for LLM and MedBERT calls — no API key or GPU is required. The preprocessor tests (`tests/test_preprocessor.py`) cover:
- Administrative header removal and placeholder stripping
- Abbreviation expansion (common conditions, medications, case insensitivity)
- Assertion detection (confirmed, negated pre/post cue, uncertain)
- Knowledge graph conflict flagging
- Priority snippet extraction from abbreviation-only notes

---

## Data

This project is designed for use with the **MIMIC IV** clinical notes dataset, available from [PhysioNet](https://physionet.org/content/mimiciv/). Access requires a credentialled PhysioNet account and completion of CITI Program training.

The `discharge` table in MIMIC IV contains free-text discharge summaries. Structured tables (`diagnoses_icd`, `prescriptions`, `procedures_icd`) can be joined by `hadm_id` and passed to the pipeline via `structured_sources`.