# Automating Structured Clinical Assessments from Discharge Notes

> **SEP-775 Final Project** — automatically extract and map free-text MIMIC IV discharge summaries to structured [MDS 3.0](https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/NursingHomeQualityInits/MDS30RAIManual) (Minimum Data Set) nursing-home assessment form fields using Large Language Models (LLMs).

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Modules](#modules)
5. [Supported MDS 3.0 Sections](#supported-mds-30-sections)
6. [Setup](#setup)
7. [GPU Support](#gpu-support)
8. [Quick Start](#quick-start)
9. [CLI Usage](#cli-usage)
10. [Configuration](#configuration)
11. [Running Tests](#running-tests)
12. [Data](#data)

---

## Overview

The system takes hospital discharge notes (unstructured free text) alongside structured clinical tables (diagnoses, medications, procedures) and automatically fills the MDS 3.0 form fields for **Section I (Active Disease Diagnoses)**, **Section N (Medications)**, and **Section O (Special Treatments, Procedures, Programs)**.

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT DATA: Discharge notes (free text) + structured tables  │
└───────────────────────────────┬──────────────────────────────┘
                                │
              ┌─────────────────┼───────────────────────┐
              │                 │                       │
              ▼                 ▼                       ▼
     ┌────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
     │ RAG BioBERT    │  │ Preprocessor    │  │ Seq2Seq preprocessor│
     │ Retriever      │  │(knowledge graph)│  │ (seq2seq input)     │
     └───────┬────────┘  └───────┬─────────┘  └──────────┬──────────┘
             │                   │                       │
             ▼                   ▼                       ▼
          ┌──────────────────────────┐       ┌─────────────────────┐
          │      LLM Extractor       │       │ Seq2Seq Extractor   │
          │      (GPT / OpenAI)      │       │ (BioBART / T5)      │
          └─────────────┬────────────┘       └──────────┬──────────┘
                        │                               │          
                        ▼                               ▼                      
                  ┌───────────┐                 ┌────────────────┐
                  │ MDS Mapper│                 │ Seq2Seq Mapper │
                  └─────┬─────┘                 └───────┬────────┘
                        │                               │                      
                        └────────────────┬──────────────┘
                                         ▼                 
                         ┌────────────────────────────────┐
                         │         MDS Assessment         │
                         │      (JSON / CSV / Excel)      │
                         └────────────────────────────────┘
```

---

## Project Structure

```
.
├── config/
│   └── config.yaml                        # Runtime configuration
├── scripts/
│   ├── check_gpu.py                       # GPU readiness diagnostic
│   ├── extract_mds_sections_from_pdf.py   # Parse MDS 3.0 PDF → item text + patterns
│   ├── preview_input_data.py              # Print sample input records
│   └── train_seq2seq.py                   # Fine-tune BioBART / ClinicalT5 on labeled pairs
├── src/
│   ├── __init__.py
│   ├── mds_schema.py                      # MDS 3.0 form field definitions (shared)
│   ├── pipeline.py                        # End-to-end orchestration
│   ├── data_preprocessor/                 # ← Load notes + preprocess text
│   │   ├── data_loader.py
│   │   ├── preprocessor.py                # Full preprocessor (LLM path)
│   │   ├── rag_retriever.py               # BioBERT semantic retriever (Option 2)
│   │   ├── seq2seq_preprocessor.py        # Simplified preprocessor (seq2seq path)
│   │   └── README.md                      # Detailed docs for this module
│   ├── extractor/                         # ← Extract MDS fields from context
│   │   ├── extractor.py                   # LLM extractor (OpenAI GPT)
│   │   ├── seq2seq_extractor.py           # Encoder-decoder extractor (BioBART / T5)
│   │   └── README.md                      # Detailed docs for this module
│   └── mapper/                            # ← Validate + map results to MDSAssessment
│       ├── mapper.py                      # Base mapper (LLM output)
│       ├── seq2seq_mapper.py              # Extended mapper (seq2seq output)
│       └── README.md                      # Detailed docs for this module
└── tests/
    ├── test_data_loader.py
    ├── test_preprocessor.py
    ├── test_extractor.py
    ├── test_mapper.py
    └── test_pipeline.py
```

---

## Pipeline Architecture

The pipeline runs three sequential stages. Each stage is self-contained and documented in its own README.

### Stage 1 — Data Preprocessor → [`src/data_preprocessor/`](src/data_preprocessor/README.md)

Loads MIMIC IV discharge notes and transforms raw clinical text into a structured **Patient Knowledge Graph** through six steps:

1. Clean & normalize text (remove de-identification placeholders, admin headers)
2. Expand clinical abbreviations (`CHF` → `congestive heart failure`)
3. Score and extract section-relevant evidence blocks (I / N / O)
4. Detect assertion status per snippet (`CONFIRMED` / `NEGATED` / `UNCERTAIN`)
5. Build knowledge graph nodes (one per evidence snippet or structured data row)
6. Detect and annotate structured ↔ unstructured conflicts

> See [src/data_preprocessor/README.md](src/data_preprocessor/README.md) for the full public API, abbreviation table, NegEx patterns, and conflict resolution logic.

---

### Stage 2 — Extractor → [`src/extractor/`](src/extractor/README.md)

Two interchangeable extractors consume the preprocessed context and return raw structured results:

| Extractor | Backend | API key required | GPU |
|-----------|---------|-----------------|-----|
| `LLMExtractor` | OpenAI Chat Completions | Yes | No |
| `Seq2SeqExtractor` | BioBART / ClinicalT5 (encoder-decoder) | No | Optional (CUDA) |

> See [src/extractor/README.md](src/extractor/README.md) for constructor parameters, GPU device selection, preprocessing modes, and fine-tuning instructions.

---

### Stage 3 — Mapper → [`src/mapper/`](src/mapper/README.md)

Validates extracted values against the MDS schema and packages them into `MDSAssessment` objects.

| Mapper | For use with |
|--------|-------------|
| `MDSMapper` | `LLMExtractor` output |
| `Seq2SeqMapper` | `Seq2SeqExtractor` output — also stores raw decoder output in metadata |

> See [src/mapper/README.md](src/mapper/README.md) for validation rules, output format, and serialisation options.

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

Sections I, N, and O are the primary extraction targets.

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

GPU acceleration is used by `Seq2SeqExtractor` (BioBART / ClinicalT5) and `BioBERTRetriever` for faster model inference and fine-tuning. The pipeline automatically detects CUDA and falls back to CPU — no code changes needed.

### Check GPU readiness

```bash
python scripts/check_gpu.py
```

**GPU available:**
```
torch_version: 2.x.x+cu118
cuda_available: True
cuda_device_count: 1
cuda_device_0: NVIDIA GeForce RTX 3080 (compute_capability=8.6)
gpu_tensor_op: PASS
```

**GPU not available:**
```
cuda_available: False
No CUDA runtime/device detected. Workloads will run on CPU.
```

Script exit codes: `0` = pass, `1` = torch import error, `2` = no CUDA, `3` = tensor op failed.

---

## Quick Start

### LLM extractor (OpenAI)

```python
from src.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline(
    source="data/discharge.csv/discharge.csv",
    openai_api_key="sk-...",
    sections=["I", "N", "O"],
    output_dir="output",
    output_format="json",
    sample_size=5,
)
assessments = pipeline.run()
```

### Seq2Seq extractor (no API key required)

```python
from src.data_preprocessor.data_loader import MIMICDischargeLoader
from src.mds_schema import MDSSchema
from src.extractor.seq2seq_extractor import Seq2SeqExtractor
from src.mapper.seq2seq_mapper import Seq2SeqMapper

loader = MIMICDischargeLoader(source="data/discharge.csv/discharge.csv")
notes = loader.load()

schema = MDSSchema(section_ids=["I", "N", "O"])
extractor = Seq2SeqExtractor(schema=schema, model_name="GanjinZero/biobart-v2-base")
mapper = Seq2SeqMapper(schema=schema)

note = notes[0]
raw = extractor.extract(note.text, note_metadata=note.metadata)
assessment = mapper.map(note.note_id, note.subject_id, note.hadm_id, raw)
print(assessment.to_dict())
```

### BioBERT RAG + LLM (semantic retrieval, no fine-tuning required)

```python
from src.data_preprocessor.data_loader import MIMICDischargeLoader
from src.data_preprocessor.rag_retriever import BioBERTRetriever
from src.mds_schema import MDSSchema
from src.extractor.extractor import LLMExtractor
from src.mapper.mapper import MDSMapper

loader = MIMICDischargeLoader(source="data/discharge.csv/discharge.csv")
notes = loader.load()

schema    = MDSSchema(section_ids=["I", "N", "O"])
retriever = BioBERTRetriever(sections=["I", "N", "O"])
extractor = LLMExtractor(schema=schema, api_key="sk-...", preprocess_input=False)
mapper    = MDSMapper(schema=schema)

note    = notes[0]
context = retriever.build_rag_context(note.text, note.metadata)
raw     = extractor.extract(context)
assessment = mapper.map(note.note_id, note.subject_id, note.hadm_id, raw)
print(assessment.to_dict())
```

### With structured data tables

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

---

## CLI Usage

```bash
python src/pipeline.py --source data/discharge.csv/discharge.csv
```

Interactive mode prompts for a processing mode:

```
Select processing mode:
  1. sample          - run the initial sample only
  2. sample-compare  - run the sample and compare preprocessing methods
  3. full            - process the entire dataset
  4. full-compare    - process the entire dataset and compare preprocessing methods
Enter mode [1-4] (default 1):
```

**Scripted / non-interactive:**

```bash
# Sample with LLM
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode sample

# Full dataset with preprocessing comparison
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode full-compare

# Custom sample size
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode sample --sample-size 20
```

---

## Configuration

Edit `config/config.yaml` to change settings without touching source code.

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

output:
  output_dir: "output"
  format: "json"
```

### Output files

| File | Contents |
|------|----------|
| `output/mds_assessments.json` | Full MDS assessment objects |
| `output/mds_ino_form_ready.json` | Compact I/N/O codes + confidence scores |
| `output/mds_ino_preprocessing_comparison.json` | Heuristic vs. LLM-evidence side-by-side |
| `output/mds_ino_preprocessing_diff_summary.json` | Fields that differ between modes |
| `output/mds_ino_preprocessing_diff_summary.csv` | Same, flattened for spreadsheet review |

---

## Running Tests

```bash
# Recommended — uses the conda environment
.conda/python.exe -m pytest tests/ -v

# Or with system Python
pytest tests/ -v
```

All tests mock LLM and Hugging Face calls — no API key or GPU required.

---

## Data

This project is designed for use with the **MIMIC IV** clinical notes dataset, available from [PhysioNet](https://physionet.org/content/mimiciv/). Access requires a credentialled PhysioNet account and completion of CITI Program training.

The `discharge` table in MIMIC IV contains free-text discharge summaries. Structured tables (`diagnoses_icd`, `prescriptions`, `procedures_icd`) can be joined by `hadm_id` and passed to the pipeline via `structured_sources`.
