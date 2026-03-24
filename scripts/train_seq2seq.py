"""
Fine-tune a seq2seq model (BioBART, ClinicalT5, SciFive) for MDS 3.0 field extraction.

The script uses the Hugging Face Trainer API with seq2seq-specific settings
(``predict_with_generate``, label smoothing, mixed-precision on GPU).

Dataset format
--------------
A CSV or JSON file with exactly two columns:

    ``text``   — raw discharge note text
    ``labels`` — MDS assessment as a compact JSON string

Example ``labels`` value::

    '{"I0200A": 1, "I2100": 0, "N0410A": 1, "O0110A1": 0}'

The keys are MDS item IDs; the values are the coded responses (int, bool, str).
The script serialises them back to a JSON string as the decoder target.

Usage
-----
Minimal::

    python scripts/train_seq2seq.py \\
        --data path/to/labeled.csv \\
        --output-dir output/seq2seq_finetuned

Full options::

    python scripts/train_seq2seq.py \\
        --data path/to/labeled.csv \\
        --model GanjinZero/biobart-v2-base \\
        --output-dir output/seq2seq_finetuned \\
        --epochs 5 \\
        --batch-size 8 \\
        --max-input-length 1024 \\
        --max-output-length 256 \\
        --val-split 0.1 \\
        --learning-rate 5e-5 \\
        --seed 42

Load the fine-tuned model afterwards::

    from src.extractor.seq2seq_extractor import Seq2SeqExtractor
    extractor = Seq2SeqExtractor(fine_tuned_path="output/seq2seq_finetuned")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running directly from the repository root
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_records(data_path: str) -> List[Dict[str, Any]]:
    """Load labeled (text, labels) pairs from CSV or JSON."""
    import pandas as pd

    path = Path(data_path)
    if path.suffix == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)

    missing = [c for c in ("text", "labels") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset '{data_path}' is missing required columns: {missing}. "
            "Expected columns: 'text' (discharge note) and 'labels' (MDS JSON string)."
        )

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        labels = row["labels"]
        if isinstance(labels, str):
            labels = json.loads(labels)
        records.append({"text": str(row["text"]), "labels": labels})

    logger.info("Loaded %d labeled records from '%s'.", len(records), data_path)
    return records


class _MDS3Dataset:
    """
    Torch-compatible dataset of (encoder_input, decoder_target) token pairs.

    Encoder input  : instruction prefix + cleaned note text
    Decoder target : compact JSON string of MDS item_id → value pairs
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: Any,
        max_input_length: int,
        max_output_length: int,
        sections: List[str],
    ) -> None:
        self._records          = records
        self._tokenizer        = tokenizer
        self._max_input        = max_input_length
        self._max_output       = max_output_length
        self._section_str      = ", ".join(sections)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import torch

        record    = self._records[idx]
        note_text = record["text"]
        label_str = json.dumps(record["labels"], ensure_ascii=False)

        input_text = (
            f"Extract MDS 3.0 sections {self._section_str} from the clinical note. "
            "Return a JSON object mapping MDS item IDs to their coded values. "
            f"Clinical note: {note_text}"
        )

        model_inputs = self._tokenizer(
            input_text,
            max_length=self._max_input,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        label_tokens = self._tokenizer(
            label_str,
            max_length=self._max_output,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        label_ids = label_tokens["input_ids"].squeeze()
        # Replace padding token id with -100 so it is ignored by the cross-entropy loss
        label_ids[label_ids == self._tokenizer.pad_token_id] = -100

        return {
            "input_ids":      model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels":         label_ids,
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    model_name: str,
    output_dir: str,
    sections: List[str],
    epochs: int,
    batch_size: int,
    max_input_length: int,
    max_output_length: int,
    val_split: float,
    learning_rate: float,
    seed: int,
) -> None:
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        logger.error(
            "Missing dependency: %s\n"
            "Install with: pip install transformers torch sentencepiece",
            exc,
        )
        sys.exit(1)

    from src.data_preprocessor.seq2seq_preprocessor import build_seq2seq_input

    # ── Load and clean records ─────────────────────────────────────────────
    records = _load_records(data_path)
    for r in records:
        r["text"] = build_seq2seq_input(
            note_text=r["text"],
            sections=sections,
            note_metadata=r.get("metadata"),
            max_chars=3400,
        )

    # ── Train / validation split ───────────────────────────────────────────
    n_val         = max(1, int(len(records) * val_split))
    val_records   = records[:n_val]
    train_records = records[n_val:]
    logger.info("Split — train: %d  val: %d", len(train_records), len(val_records))

    # ── Tokenizer and model ────────────────────────────────────────────────
    logger.info("Loading model '%s'…", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_dataset = _MDS3Dataset(
        train_records, tokenizer, max_input_length, max_output_length, sections
    )
    val_dataset = _MDS3Dataset(
        val_records, tokenizer, max_input_length, max_output_length, sections
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # ── Training arguments ─────────────────────────────────────────────────
    use_fp16 = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        predict_with_generate=True,
        generation_max_length=max_output_length,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=use_fp16,
        seed=seed,
        logging_steps=10,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    logger.info("Starting fine-tuning (fp16=%s)…", use_fp16)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Fine-tuned model saved to '%s'.", output_dir)
    logger.info(
        "Load it with: Seq2SeqExtractor(fine_tuned_path='%s')", output_dir
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fine-tune a seq2seq model (BioBART, ClinicalT5, SciFive) "
            "for MDS 3.0 field extraction from discharge notes."
        )
    )
    p.add_argument(
        "--data", required=True,
        help="Path to labeled CSV or JSON dataset (columns: 'text', 'labels').",
    )
    p.add_argument(
        "--model", default="GanjinZero/biobart-v2-base",
        help="Hugging Face model ID. Default: GanjinZero/biobart-v2-base.",
    )
    p.add_argument(
        "--output-dir", default="output/seq2seq_finetuned",
        help="Directory to save the fine-tuned checkpoint.",
    )
    p.add_argument(
        "--sections", nargs="+", default=["I", "N", "O"],
        help="MDS sections to include in the task prompt. Default: I N O.",
    )
    p.add_argument("--epochs",            type=int,   default=5)
    p.add_argument("--batch-size",        type=int,   default=8)
    p.add_argument("--max-input-length",  type=int,   default=1024,
                   help="Encoder token budget (default 1024, BART max).")
    p.add_argument("--max-output-length", type=int,   default=256,
                   help="Decoder generation limit in tokens.")
    p.add_argument(
        "--val-split", type=float, default=0.1,
        help="Fraction of data held out for validation (default 0.1).",
    )
    p.add_argument("--learning-rate",     type=float, default=5e-5)
    p.add_argument("--seed",              type=int,   default=42)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(
        data_path        = args.data,
        model_name       = args.model,
        output_dir       = args.output_dir,
        sections         = args.sections,
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        max_input_length = args.max_input_length,
        max_output_length= args.max_output_length,
        val_split        = args.val_split,
        learning_rate    = args.learning_rate,
        seed             = args.seed,
    )
