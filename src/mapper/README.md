# Mapper

This package validates raw extraction results against the MDS 3.0 schema and packages them into `MDSAssessment` objects that can be serialised to JSON, CSV, or Excel.

## Files

| File | Responsibility |
|------|---------------|
| `mapper.py` | Base mapper for LLM extraction results |
| `seq2seq_mapper.py` | Extended mapper for Seq2Seq extraction results — stores raw model output in metadata |

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

## seq2seq_mapper.py — `Seq2SeqMapper`

Extends `MDSMapper` with additional handling for Seq2Seq-specific output keys. In addition to standard validation, it:

- Strips `_raw_output` and `_model_name` from the extraction dict before base validation
- Persists them into `assessment.metadata` under `seq2seq_raw_output` and `seq2seq_model`

### Usage

```python
from src.extractor.seq2seq_extractor import Seq2SeqExtractor
from src.mapper.seq2seq_mapper import Seq2SeqMapper
from src.mds_schema import MDSSchema

schema = MDSSchema(section_ids=["I", "N", "O"])

extractor = Seq2SeqExtractor(schema=schema)
mapper = Seq2SeqMapper(schema=schema)

raw = extractor.extract(note.text, note_metadata=note.metadata)
assessment = mapper.map(
    note_id=note.note_id,
    subject_id=note.subject_id,
    hadm_id=note.hadm_id,
    extraction=raw,
)

# Access Seq2Seq-specific metadata
print(assessment.metadata["seq2seq_model"])
# → "GanjinZero/biobart-v2-base"

print(assessment.metadata["seq2seq_raw_output"])
# → '{"I0200A": 1, "N0410A": 1, ...}'
```

### Extra metadata keys

| Key | Type | Description |
|-----|------|-------------|
| `extractor` | `str` | Always `"seq2seq"` |
| `seq2seq_model` | `str` | Hugging Face model name used for extraction |
| `seq2seq_raw_output` | `str` | First 1,000 chars of the raw decoder output string |

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
.conda/python.exe -m pytest tests/test_mapper.py -v
```

All tests use mocked extraction output — no API key or GPU needed.
