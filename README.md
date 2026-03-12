# Automating Structured Clinical Assessments from Discharge Notes

> **SEP-775 Final Project** — automatically extract and map free-text MIMIC IV discharge summaries to structured [MDS 3.0](https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/NursingHomeQualityInits/MDS30RAIManual) (Minimum Data Set) nursing-home assessment form fields using Large Language Models (LLMs).

---

## Overview

The system takes unstructured hospital discharge notes (stored in Excel/CSV files or loaded directly from the [MIMIC IV](https://physionet.org/content/mimiciv/) dataset via [PyHealth](https://pyhealth.readthedocs.io/)) and automatically fills the clinically relevant sections of the MDS 3.0 form.

```
Discharge Note (free text)
        │
        ▼
 MIMICDischargeLoader   ← Excel / CSV / PyHealth MIMIC IV
        │
        ▼
   LLMExtractor         ← GPT-3.5-Turbo (OpenAI) structured prompting
        │
        ▼
    MDSMapper           ← Validates & maps to MDS 3.0 schema
        │
        ▼
  MDSAssessment         ← JSON / CSV / Excel output
```

---

## Project Structure

```
.
├── config/
│   └── config.yaml          # Runtime configuration
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Load discharge notes (Excel/CSV or PyHealth)
│   ├── mds_schema.py        # MDS 3.0 form field definitions
│   ├── extractor.py         # LLM-based field extraction
│   ├── mapper.py            # Validate and map extracted values
│   └── pipeline.py          # End-to-end orchestration
├── tests/
│   ├── test_data_loader.py
│   ├── test_mds_schema.py
│   ├── test_extractor.py
│   ├── test_mapper.py
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

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
| I | Active Disease Diagnoses |
| J | Health Conditions (Pain, Dyspnea, Falls) |
| K | Swallowing / Nutritional Status |
| M | Skin Conditions / Pressure Ulcers |
| N | Medications |
| O | Special Treatments, Procedures, Programs |

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

## Usage

### Quick start — from an Excel file

```python
from src.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline(
    source="data/discharge_notes.xlsx",   # path to your Excel file
    openai_api_key="sk-...",              # or set OPENAI_API_KEY env var
    sections=["I", "J", "G"],            # MDS sections to extract
    output_dir="output",
    output_format="json",                 # "json", "csv", or "excel"
)
assessments = pipeline.run()
```

### Loading from MIMIC IV via PyHealth

```python
from src.pipeline import ExtractionPipeline

pipeline = ExtractionPipeline(
    source="pyhealth",
    mimic_root="/path/to/mimic-iv",       # your MIMIC IV data directory
    openai_api_key="sk-...",
    output_dir="output",
    output_format="csv",
)
assessments = pipeline.run()
```

### Using individual modules

```python
from src.data_loader import MIMICDischargeLoader
from src.mds_schema import MDSSchema
from src.extractor import LLMExtractor
from src.mapper import MDSMapper

# 1. Load notes
loader = MIMICDischargeLoader(source="data/discharge_notes.xlsx")
notes = loader.load()

# 2. Set up extractor (only sections I and J)
schema = MDSSchema()
extractor = LLMExtractor(schema=schema, sections=["I", "J"])
mapper = MDSMapper(schema=schema)

# 3. Process the first note
note = notes[0]
raw = extractor.extract(note.text)
assessment = mapper.map(note.note_id, note.subject_id, note.hadm_id, raw)

print(assessment.to_dict())
```

### Input Excel / CSV format

The input file must contain at least these four columns (names are configurable):

| note_id | subject_id | hadm_id | text |
|---------|------------|---------|------|
| 1001    | 10001      | 200001  | Patient is a 78-year-old male … |

---

## Configuration

Edit `config/config.yaml` to change LLM settings, data paths, and output options without touching source code.

Key settings:

```yaml
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  temperature: 0.0

data:
  excel_path: "data/discharge_notes.xlsx"

output:
  output_dir: "output"
  format: "json"
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests use mocks for the LLM calls — no API key is required.

---

## Data

This project is designed for use with the **MIMIC IV** clinical notes dataset, available from [PhysioNet](https://physionet.org/content/mimiciv/). Access requires a credentialled PhysioNet account and completion of the required training.

The `discharge` table contains free-text discharge summaries which serve as the primary input to this pipeline.
