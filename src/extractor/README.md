# Extractor

This package provides extractors that consume a **Patient Knowledge Graph** (produced by `data_preprocessor`) and return raw structured extraction results for the mapper.

## Files

| File | Responsibility |
|------|---------------|
| `extractor.py` | LLM-based extraction via OpenAI Chat Completions |
| `seq2seq_extractor.py` | Encoder-decoder extraction using BioBART / ClinicalT5 |

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

## seq2seq_extractor.py — `Seq2SeqExtractor`

Uses a fine-tuned BioBART or ClinicalT5 encoder-decoder model to generate MDS JSON directly from the note context. No OpenAI API key is required.

The simplified seq2seq preprocessor (`build_seq2seq_input`) is called internally to truncate the note to fit within the model's context window (≤ 1,024 tokens for BART, ≤ 512 tokens for T5).

### Usage

```python
from src.extractor.seq2seq_extractor import Seq2SeqExtractor
from src.mds_schema import MDSSchema

schema = MDSSchema(section_ids=["I", "N", "O"])

extractor = Seq2SeqExtractor(
    schema=schema,
    model_name="GanjinZero/biobart-v2-base",  # or a fine-tuned checkpoint
    sections=["I", "N", "O"],
    prefer_gpu=True,
    gpu_device=0,
)

raw = extractor.extract(note.text, note_metadata=note.metadata)
# raw → {item_id: value, "confidence": {item_id: float}, "_raw_output": str}
```

### Constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `schema` | required | `MDSSchema` instance |
| `model_name` | `"GanjinZero/biobart-v2-base"` | Hugging Face model ID or local checkpoint path |
| `sections` | `["I","N","O"]` | MDS sections to extract |
| `prefer_gpu` | `True` | Use CUDA when available |
| `gpu_device` | `0` | CUDA device index |
| `max_input_length` | `1024` | Token limit for the encoder input |
| `max_output_length` | `512` | Token limit for the decoder output |
| `num_beams` | `4` | Beam search width |
| `fine_tuned_path` | `None` | Path to a fine-tuned checkpoint (overrides `model_name`) |

### GPU device selection

| Scenario | Log message |
|---|---|
| GPU found | `Seq2SeqExtractor using GPU (CUDA device 0: NVIDIA GeForce RTX 3080).` |
| Requested device out of range | `Requested CUDA device N unavailable (device_count=M). Falling back to 0.` |
| GPU preference disabled | `Seq2SeqExtractor using CPU (GPU preference disabled).` |
| CUDA not available | `Seq2SeqExtractor using CPU (CUDA not available).` |
| `torch` import failed | `Seq2SeqExtractor using CPU (torch import failed: …).` |

### Fine-tuning

Use `scripts/train_seq2seq.py` to fine-tune on labeled (note → MDS JSON) pairs:

```bash
.conda/python.exe scripts/train_seq2seq.py \
    --data data/labeled_pairs.csv \
    --output-dir models/biobart-finetuned \
    --epochs 5
```

See `scripts/train_seq2seq.py --help` for all options.

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
.conda/python.exe -m pytest tests/test_extractor.py -v
```

All tests mock LLM and Hugging Face calls — no API key or GPU is needed.
