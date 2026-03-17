# Automating Structured Clinical Assessments from Discharge Notes

> **SEP-775 Final Project** — automatically extract and map free-text MIMIC IV discharge summaries to structured [MDS 3.0](https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/NursingHomeQualityInits/MDS30RAIManual) (Minimum Data Set) nursing-home assessment form fields using Large Language Models (LLMs).

---

## Overview

The system takes unstructured hospital discharge notes (stored in Excel/CSV files or loaded directly from the [MIMIC IV](https://physionet.org/content/mimiciv/) dataset via [PyHealth](https://pyhealth.readthedocs.io/)) and automatically fills the clinically relevant sections of the MDS 3.0 form.

The default workflow is now focused on **Section I (Diagnoses), Section N (Medications), and Section O (Special Treatments/Procedures)**:
1. preprocess discharge text (remove noise/placeholders and surface I/N/O evidence),
2. run extraction with either:
  - **GPT/OpenAI LLM** (`LLMExtractor`), or
  - **MedBERT biomedical model** (`MedBERTExtractor`, Hugging Face token classification),
3. produce JSON that can be used directly to fill MDS I/N/O fields.

```
Discharge Note (free text)
        │
        ▼
 MIMICDischargeLoader   ← Excel / CSV / PyHealth MIMIC IV
        │
        ▼
   DischargePreprocessor  ← clean + focus text for I/N/O evidence
     │
     ├──────────────► LLMExtractor      ← GPT-3.5-Turbo (OpenAI) structured prompting
     │
     └──────────────► MedBERTExtractor  ← Biomedical NER + rule-based item mapping
        │
        ▼
  MDSMapper / MedBERTMapper  ← Validates & maps to MDS 3.0 schema
        │
        ▼
  MDSAssessment         ← JSON / CSV / Excel output
```

---

## Project Structure

```
.
├── config/
│   ├── config.yaml                  # Runtime configuration
│   └── medbert_patterns.auto.json   # Auto-generated MedBERT regex seed patterns
├── scripts/
│   ├── extract_mds_sections_from_pdf.py  # Parse MDS 3.0 PDF into section/item text + patterns
│   └── preview_input_data.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Load discharge notes (Excel/CSV or PyHealth)
│   ├── mds_schema.py        # MDS 3.0 form field definitions
│   ├── extractor.py         # LLM-based field extraction
│   ├── medbert_extractor.py # MedBERT-based biomedical NER extraction
│   ├── mapper.py            # Validate and map extracted values
│   ├── medbert_mapper.py    # Validate/map MedBERT outputs to schema
│   ├── preprocessor.py      # Shared note cleaning + evidence-focused context builder
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
  source="data/discharge.csv/discharge.csv",  # path to your discharge csv
    openai_api_key="sk-...",              # or set OPENAI_API_KEY env var
  sections=["I", "N", "O"],            # target sections
    output_dir="output",
    output_format="json",                 # "json", "csv", or "excel"
    sample_size=5,                         # default sample-first run
)
assessments = pipeline.run()
```

By default, the pipeline runs a small sample first. To process the entire input dataset, set `process_all_notes=True` in Python or choose a `full` mode when prompted by the CLI.

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
from src.medbert_extractor import MedBERTExtractor
from src.medbert_mapper import MedBERTMapper

# 1. Load notes
loader = MIMICDischargeLoader(source="data/discharge.csv/discharge.csv")
notes = loader.load()

# 2. Set up extractor (sections I/N/O)
schema = MDSSchema()
extractor = LLMExtractor(schema=schema, sections=["I", "N", "O"])
mapper = MDSMapper(schema=schema)

# 3. Process the first note
note = notes[0]
raw = extractor.extract(note.text)
assessment = mapper.map(note.note_id, note.subject_id, note.hadm_id, raw)

print(assessment.to_dict())
```

### MedBERT extractor (biomedical model + rule mapping)

The GPT extractor is kept unchanged. A second extractor/mapper pair is now available:

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
  model_name="d4data/biomedical-ner-all",  # replace with your MedBERT checkpoint
  sections=["I", "N", "O"],
)
mapper = MedBERTMapper(schema=schema)

note = notes[0]
raw = extractor.extract(note.text)
assessment = mapper.map(note.note_id, note.subject_id, note.hadm_id, raw)

print(assessment.to_dict())
```

`MedBERTMapper` stores NER entities and evidence snippets in `assessment.metadata` for auditability.

## MDS 3.0 form data extraction (from official PDF)

The project includes a utility that parses the MDS 3.0 reference form pages for Sections I/N/O and exports machine-readable item text. This is used to improve pattern quality and keep mapping aligned to form wording.

Run:

```bash
python scripts/extract_mds_sections_from_pdf.py
```

Generated outputs:
- [output/mds_sections_from_pdf.generated.json](output/mds_sections_from_pdf.generated.json)
- [config/medbert_patterns.auto.json](config/medbert_patterns.auto.json)

### Auto-generate MedBERT patterns from MDS 3.0 PDF

To reduce manual pattern maintenance, generate regex seed patterns directly from the MDS PDF pages:

```bash
python scripts/extract_mds_sections_from_pdf.py
```

This reads:
- Section I: pages 29–31
- Section N: pages 42–43
- Section O: pages 44–49

and writes [config/medbert_patterns.auto.json](config/medbert_patterns.auto.json).

`MedBERTExtractor` will automatically load this file (if present) and merge only label-compatible patterns as safe overrides.

### Input Excel / CSV format

The input file must contain at least these four columns (names are configurable):

| note_id | subject_id | hadm_id | text |
|---------|------------|---------|------|
| 1001    | 10001      | 200001  | Patient is a 78-year-old male … |

---

## Configuration

Edit `config/config.yaml` to change LLM settings, data paths, preprocessing, and output options without touching source code.

Key settings:

```yaml
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  temperature: 0.0

data:
  csv_path: "data/discharge.csv/discharge.csv"

preprocess:
  enabled: true

output:
  output_dir: "output"
  format: "json"

# Additional output when format=json:
# output/mds_ino_form_ready.json
# output/mds_ino_preprocessing_comparison.json
# output/mds_ino_preprocessing_diff_summary.json
# output/mds_ino_preprocessing_diff_summary.csv
```

### CLI — interactive mode prompt

When run from a terminal, the pipeline always asks which processing mode you want before starting:

```
$ python src/pipeline.py --source data/discharge.csv/discharge.csv

Select processing mode:
  1. sample          - run the initial sample only
  2. sample-compare  - run the sample and compare preprocessing methods
  3. full            - process the entire dataset
  4. full-compare    - process the entire dataset and compare preprocessing methods
  5. medbert-sample  - run the initial sample with MedBERT NER
  6. medbert-full    - process the entire dataset with MedBERT NER
Enter mode [1-6] (default 1):
```

Enter a number (1–6) or a mode name, or press **Enter** to accept the default (`sample`).

| Mode | Notes processed | Comparison output |
|------|-----------------|-------------------|
| `sample` | First `--sample-size` notes (default 5) | No |
| `sample-compare` | First `--sample-size` notes | Yes — diff CSV + JSON |
| `full` | All notes | No |
| `full-compare` | All notes | Yes — diff CSV + JSON |
| `medbert-sample` | First `--sample-size` notes | No (MedBERT backend) |
| `medbert-full` | All notes | No (MedBERT backend) |

### CLI — scripted / non-interactive use

Pass `--mode` to skip the prompt entirely (e.g. in automated pipelines):

```bash
# Sample run, no comparison
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode sample

# Sample run with preprocessing comparison
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode sample-compare

# Full dataset
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode full

# Full dataset with comparison
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode full-compare

# Sample run using MedBERT backend
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode medbert-sample

# Full dataset using MedBERT backend
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode medbert-full
```

You can override the sample size with `--sample-size`:

```bash
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode sample --sample-size 20
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

# Automating Structured Clinical Assessments from Discharge Notes

> **SEP-775 Final Project** — automatically extract and map free-text MIMIC IV discharge summaries to structured [MDS 3.0](https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/NursingHomeQualityInits/MDS30RAIManual) (Minimum Data Set) nursing-home assessment form fields using Large Language Models (LLMs).

---

## Overview

The system takes unstructured hospital discharge notes (stored in Excel/CSV files or loaded directly from the [MIMIC IV](https://physionet.org/content/mimiciv/) dataset via [PyHealth](https://pyhealth.readthedocs.io/)) and automatically fills the clinically relevant sections of the MDS 3.0 form.

The default workflow is now focused on **Section I (Diagnoses), Section N (Medications), and Section O (Special Treatments/Procedures)**:
1. preprocess discharge text (remove noise/placeholders and surface I/N/O evidence),
2. run LLM extraction with MDS-coded outputs,
3. produce JSON that can be used directly to fill MDS I/N/O fields.

```
Discharge Note (free text)
        │
        ▼
 MIMICDischargeLoader   ← Excel / CSV / PyHealth MIMIC IV
        │
        ▼
   DischargePreprocessor  ← clean + focus text for I/N/O evidence
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
  source="data/discharge.csv/discharge.csv",  # path to your discharge csv
    openai_api_key="sk-...",              # or set OPENAI_API_KEY env var
  sections=["I", "N", "O"],            # target sections
    output_dir="output",
    output_format="json",                 # "json", "csv", or "excel"
    sample_size=5,                         # default sample-first run
)
assessments = pipeline.run()
```

By default, the pipeline runs a small sample first. To process the entire input dataset, set `process_all_notes=True` in Python or choose a `full` mode when prompted by the CLI.

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
loader = MIMICDischargeLoader(source="data/discharge.csv/discharge.csv")
notes = loader.load()

# 2. Set up extractor (sections I/N/O)
schema = MDSSchema()
extractor = LLMExtractor(schema=schema, sections=["I", "N", "O"])
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

Edit `config/config.yaml` to change LLM settings, data paths, preprocessing, and output options without touching source code.

Key settings:

```yaml
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  temperature: 0.0

data:
  csv_path: "data/discharge.csv/discharge.csv"

preprocess:
  enabled: true

output:
  output_dir: "output"
  format: "json"

# Additional output when format=json:
# output/mds_ino_form_ready.json
# output/mds_ino_preprocessing_comparison.json
# output/mds_ino_preprocessing_diff_summary.json
# output/mds_ino_preprocessing_diff_summary.csv
```

### CLI — interactive mode prompt

When run from a terminal, the pipeline always asks which processing mode you want before starting:

```
$ python src/pipeline.py --source data/discharge.csv/discharge.csv

Select processing mode:
  1. sample          - run the initial sample only
  2. sample-compare  - run the sample and compare preprocessing methods
  3. full            - process the entire dataset
  4. full-compare    - process the entire dataset and compare preprocessing methods
Enter mode [1-4] (default 1):
```

Enter a number (1–4) or a mode name, or press **Enter** to accept the default (`sample`).

| Mode | Notes processed | Comparison output |
|------|-----------------|-------------------|
| `sample` | First `--sample-size` notes (default 5) | No |
| `sample-compare` | First `--sample-size` notes | Yes — diff CSV + JSON |
| `full` | All notes | No |
| `full-compare` | All notes | Yes — diff CSV + JSON |

### CLI — scripted / non-interactive use

Pass `--mode` to skip the prompt entirely (e.g. in automated pipelines):

```bash
# Sample run, no comparison
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode sample

# Sample run with preprocessing comparison
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode sample-compare

# Full dataset
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode full

# Full dataset with comparison
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode full-compare
```

You can override the sample size with `--sample-size`:

```bash
python src/pipeline.py --source data/discharge.csv/discharge.csv --mode sample --sample-size 20
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
