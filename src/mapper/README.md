# Mapper

This package validates raw extraction results against the MDS 3.0 schema and packages them into `MDSAssessment` objects that can be serialised to JSON, CSV, or Excel.

## Files

| File | Responsibility |
|------|---------------|
| `mapper.py` | Base mapper for LLM extraction results |
| `medbert_mapper.py` | Extended mapper for MedBERT extraction results — adds evidence trace metadata |

---

## mapper.py — `MDSMapper`

Validates each extracted value against the MDS schema, discards values that fail validation (logging a warning for each), and stores the remaining values with per-item confidence scores in an `MDSAssessment`.

### Validation rules

| Item type | Validation |
|-----------|-----------|
| `BOOLEAN` | Must be `True` / `False` / `0` / `1` |
| `INTEGER` | Must be a valid integer; optionally checked against allowed values |
| `SELECT` | Code must appear in the item's `response_options` list |
| `MULTI_SELECT` | Each code must appear in the item's `response_options` list |
| `TEXT` | Any non-empty string accepted |

### Usage

```python
from src.mapper.mapper import MDSMapper
from src.mds_schema import MDSSchema

schema = MDSSchema(section_ids=["I", "N", "O"])
mapper = MDSMapper(schema=schema)

# raw is the dict returned by LLMExtractor.extract()
assessment = mapper.map(
    note_id="1001",
    subject_id="10001",
    hadm_id="200001",
    extraction=raw,
)

print(assessment.to_dict())
```

### `map()` parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `note_id` | `str` | Discharge note identifier |
| `subject_id` | `str` | Patient identifier |
| `hadm_id` | `str` | Hospital admission identifier |
| `extraction` | `dict` | Raw extraction output from `LLMExtractor.extract()` |

### `MDSAssessment` output methods

```python
assessment.to_dict()    # Full assessment as a Python dict
assessment.to_json()    # JSON string
assessment.to_csv()     # Flattened CSV string
```

Fields included per item:

| Field | Description |
|-------|-------------|
| `value` | The validated coded value |
| `confidence` | Float 0–1 confidence score from the extractor |
| `evidence` | Supporting text snippet cited by the extractor |
| `status` | `"valid"` or `"invalid"` |

### Constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `schema` | `None` | `MDSSchema` instance — uses default schema if omitted |
| `strict` | `False` | If `True`, raise on invalid values instead of discarding |

---

## medbert_mapper.py — `MedBERTMapper`

Extends `MDSMapper` with additional handling for MedBERT-specific output keys. In addition to standard validation, it:

- Persists `_evidence` (entity text → item ID mapping) into `assessment.metadata["medbert_evidence"]`
- Persists `_entities` (raw NER entity list) into `assessment.metadata["medbert_entities"]`

### Usage

```python
from src.extractor.medbert_extractor import MedBERTExtractor
from src.mapper.medbert_mapper import MedBERTMapper
from src.mds_schema import MDSSchema

schema = MDSSchema(section_ids=["I", "N", "O"])

extractor = MedBERTExtractor(schema=schema)
mapper = MedBERTMapper(schema=schema)

raw = extractor.extract(note.text, note_metadata=note.metadata)
assessment = mapper.map(
    note_id=note.note_id,
    subject_id=note.subject_id,
    hadm_id=note.hadm_id,
    extraction=raw,
)

# Access MedBERT-specific metadata
print(assessment.metadata["medbert_evidence"])
# → {"I0200A": ["heart failure", "CHF"], "N0410A": ["warfarin 5mg"], ...}

print(assessment.metadata["medbert_entities"])
# → [{"word": "heart failure", "entity_group": "Disease", "score": 0.98, ...}, ...]
```

### Extra metadata keys

| Key | Type | Description |
|-----|------|-------------|
| `medbert_evidence` | `dict[str, list[str]]` | Entity text spans that triggered each item ID |
| `medbert_entities` | `list[dict]` | Full NER entity output from Hugging Face pipeline |

---

## Complete extraction → mapping example

```python
from src.data_preprocessor.data_loader import MIMICDischargeLoader
from src.mds_schema import MDSSchema
from src.extractor.extractor import LLMExtractor
from src.mapper.mapper import MDSMapper

loader = MIMICDischargeLoader(source="data/discharge.csv/discharge.csv")
notes = loader.load()

schema = MDSSchema(section_ids=["I", "N", "O"])
extractor = LLMExtractor(schema=schema, api_key="sk-...")
mapper = MDSMapper(schema=schema)

note = notes[0]
raw = extractor.extract(note.text, note_metadata=note.metadata)
assessment = mapper.map(note.note_id, note.subject_id, note.hadm_id, raw)

print(assessment.to_dict())
```

---

## Running tests

```bash
.conda/python.exe -m pytest tests/test_mapper.py tests/test_medbert_mapper.py -v
```

All tests use mocked extraction output — no API key or GPU needed.
