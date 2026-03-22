# Extractor

This package provides two independent extractors that consume a **Patient Knowledge Graph** (produced by `data_preprocessor`) and return raw structured extraction results for the mapper.

## Files

| File | Responsibility |
|------|---------------|
| `extractor.py` | LLM-based extraction via OpenAI Chat Completions |
| `medbert_extractor.py` | Biomedical NER extraction using a Hugging Face token-classification model |

Both extractors share the same output format so they can feed the same mapper.

---

## extractor.py — `LLMExtractor`

Sends the preprocessed knowledge graph context to an OpenAI GPT model with a structured system prompt, and returns a JSON mapping of MDS item IDs to coded values.

### Supported preprocessing modes

| Mode | What it does |
|------|-------------|
| `heuristic` (default) | Uses `build_extraction_context()` from the preprocessor — rule-based evidence selection |
| `llm_evidence` | Uses the LLM itself for a lightweight first-pass condensation before final extraction |

### Usage

```python
from src.extractor.extractor import LLMExtractor
from src.mds_schema import MDSSchema

schema = MDSSchema(section_ids=["I", "N", "O"])

extractor = LLMExtractor(
    schema=schema,
    api_key="sk-...",          # or set OPENAI_API_KEY env var
    model="gpt-3.5-turbo",     # any OpenAI Chat Completions model
    temperature=0.0,
    preprocessing_mode="heuristic",
)

raw = extractor.extract(note.text, note_metadata=note.metadata)
# raw → dict[item_id → {"value": ..., "confidence": float, "evidence": str}]
```

### Constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `schema` | required | `MDSSchema` instance defining which items to extract |
| `api_key` | `None` | OpenAI API key — falls back to `OPENAI_API_KEY` env var |
| `model` | `"gpt-3.5-turbo"` | OpenAI model name |
| `temperature` | `0.0` | Sampling temperature (0.0 = deterministic) |
| `max_tokens` | `2048` | Maximum tokens in the LLM response |
| `preprocessing_mode` | `"heuristic"` | `"heuristic"` or `"llm_evidence"` |
| `sections` | `["I","N","O"]` | MDS sections to extract |

### Retry behaviour

`LLMExtractor` uses `tenacity` to retry transient API failures with exponential back-off (up to 3 attempts by default). Rate-limit and server errors are retried; parsing errors are not.

### Setting the API key

```bash
# Shell environment variable (preferred)
export OPENAI_API_KEY="sk-..."

# Or pass directly
extractor = LLMExtractor(schema=schema, api_key="sk-...")
```

---

## medbert_extractor.py — `MedBERTExtractor`

Uses a Hugging Face token-classification checkpoint to run biomedical named entity recognition (NER) over the preprocessed note. Identified entity spans are matched to MDS item IDs using regex patterns generated from the MDS 3.0 PDF.

No OpenAI API key is required.

### Usage

```python
from src.extractor.medbert_extractor import MedBERTExtractor
from src.mds_schema import MDSSchema

schema = MDSSchema(section_ids=["I", "N", "O"])

extractor = MedBERTExtractor(
    schema=schema,
    model_name="d4data/biomedical-ner-all",  # any HF token-classification model
    sections=["I", "N", "O"],
    prefer_gpu=True,   # default: True — uses CUDA when available
    gpu_device=0,      # CUDA device index
)

raw = extractor.extract(note.text, note_metadata=note.metadata)
```

### Constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `schema` | required | `MDSSchema` instance |
| `model_name` | `"d4data/biomedical-ner-all"` | Hugging Face model ID |
| `sections` | `["I","N","O"]` | MDS sections to extract |
| `prefer_gpu` | `True` | Use CUDA when available |
| `gpu_device` | `0` | CUDA device index |
| `patterns_path` | auto | Path to `medbert_patterns.auto.json` |

### GPU device selection

The extractor resolves the device at model load time and logs the outcome:

| Scenario | Log message |
|---|---|
| GPU found | `MedBERT using GPU (CUDA device 0: NVIDIA GeForce RTX 3080).` |
| Requested device out of range | `Requested CUDA device N is unavailable (device_count=M). Falling back to 0.` |
| GPU preference disabled | `MedBERT using CPU (GPU preference disabled).` |
| CUDA not available | `MedBERT using CPU (CUDA not available).` |
| No CUDA devices | `MedBERT using CPU (no CUDA devices detected).` |
| `torch` import failed | `MedBERT using CPU (torch import failed: …).` |

Force CPU explicitly:

```python
extractor = MedBERTExtractor(schema=schema, prefer_gpu=False)
```

Target a specific GPU in a multi-GPU machine:

```python
extractor = MedBERTExtractor(schema=schema, prefer_gpu=True, gpu_device=1)
```

### Extraction internals

1. The preprocessor's `build_extraction_context()` is called to focus the note text.
2. The Hugging Face `pipeline("ner", ...)` runs token classification, grouping sub-word tokens into entity spans.
3. Entity labels (e.g. `B-Disease`, `B-Chemical`) are mapped to MDS sections (I / N / O).
4. Each entity is matched against section-specific regex patterns (`config/medbert_patterns.auto.json`) to determine the MDS item ID.
5. Results include a `_evidence` trace (entity text → item ID) and an `_entities` list (full NER output).

### Checking GPU readiness before inference

```bash
python scripts/check_gpu.py
```

Expected output when GPU is available:
```
torch_version: 2.x.x+cu118
cuda_available: True
cuda_device_count: 1
cuda_device_0: NVIDIA GeForce RTX 3080 (compute_capability=8.6)
gpu_tensor_op: PASS
```

---

## Running tests

```bash
.conda/python.exe -m pytest tests/test_extractor.py tests/test_medbert_extractor.py -v
```

All tests mock LLM and Hugging Face calls — no API key or GPU is needed.
