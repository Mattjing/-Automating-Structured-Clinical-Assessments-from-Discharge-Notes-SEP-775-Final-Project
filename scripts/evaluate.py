"""
evaluate.py
===========
Evaluates model output against ground-truth MDS 3.0 labels.

Usage
-----
    python scripts/evaluate.py \\
        --predictions output/mds_assessments.json \\
        --ground-truth data/test/extraction_ground_truth.json

Output includes:
  - Per-field precision / recall / F1
  - Per-section (I, N, O) summary
  - Overall micro-averaged F1
  - A CSV report written to output/evaluation_report.csv

Supported ground-truth formats
-------------------------------
1. **Flat extraction format** (extraction_ground_truth.json) — MDS item IDs
   are top-level keys alongside ``note_id`` / ``subject_id`` / ``hadm_id``
   and a nested ``"confidence"`` dict.  This is the exact format consumed by
   ``MDSMapper.map()``.
2. **Nested fields format** — each record has a ``"fields"`` or ``"codes"``
   key containing the MDS item dict (as produced by the pipeline output files
   ``mds_assessments.json`` and ``mds_ino_form_ready.json``).

The evaluator auto-detects the format.

Matching strategy
-----------------
  - Records are matched by ``note_id`` (or ``subject_id + hadm_id`` fallback).
  - Only fields present in the ground truth are evaluated; model-predicted
    fields absent from ground truth count as false positives.
  - BOOLEAN: predicted truthy value vs. ground-truth boolean.
  - INTEGER: exact match.
  - SELECT / TEXT: exact string match after lower-casing.
  - MULTI_SELECT: set-equality of code lists.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(value: Any) -> Any:
    """Normalise a field value for comparison."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no"):
            return False
        return v
    if isinstance(value, list):
        return set(str(c).strip() for c in value)
    return value


def _values_match(predicted: Any, expected: Any) -> bool:
    p = _normalize(predicted)
    e = _normalize(expected)
    if isinstance(e, set) and isinstance(p, set):
        return p == e
    if isinstance(e, set):
        return {str(p)} == e
    if isinstance(p, set):
        return p == {str(e)}
    return p == e


def _is_positive(value: Any) -> bool:
    """Return True if value represents a clinically positive/active finding."""
    if value is None:
        return False
    v = _normalize(value)
    if isinstance(v, bool):
        return v
    if isinstance(v, set):
        return bool(v)
    if isinstance(v, int):
        return v > 0
    if isinstance(v, str):
        return v not in ("false", "0", "no", "none", "")
    return bool(v)


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------


def load_json_records(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def build_index(records: List[Dict]) -> Dict[str, Dict]:
    """Index records by note_id (primary) or subject_id+hadm_id (fallback)."""
    index: Dict[str, Dict] = {}
    for rec in records:
        key = rec.get("note_id") or f"{rec.get('subject_id','')}-{rec.get('hadm_id','')}"
        index[str(key)] = rec
    return index


# ---------------------------------------------------------------------------
# Per-field evaluation
# ---------------------------------------------------------------------------


FieldResult = Dict[str, Any]  # {field_id, tp, fp, fn, tn, match}


def evaluate_pair(
    predicted_fields: Dict[str, Any],
    ground_truth_fields: Dict[str, Any],
) -> List[FieldResult]:
    """
    Compare predicted fields against ground truth for one patient record.

    Returns a list of per-field result dicts.
    """
    results: List[FieldResult] = []
    all_field_ids = set(ground_truth_fields) | set(predicted_fields)

    for fid in sorted(all_field_ids):
        gt_val = ground_truth_fields.get(fid)
        pred_val = predicted_fields.get(fid)

        gt_positive = _is_positive(gt_val)
        pred_positive = _is_positive(pred_val)
        exact_match = (fid in predicted_fields) and _values_match(pred_val, gt_val)

        results.append(
            {
                "field_id": fid,
                "section": fid[0] if fid else "?",
                "ground_truth": gt_val,
                "predicted": pred_val,
                "exact_match": exact_match,
                "tp": int(gt_positive and pred_positive),
                "fp": int((not gt_positive) and pred_positive),
                "fn": int(gt_positive and (not pred_positive)),
                "tn": int((not gt_positive) and (not pred_positive)),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def aggregate_metrics(all_results: List[FieldResult]) -> Dict[str, Any]:
    """Compute micro-averaged and per-section metrics."""
    totals = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "exact_match": 0, "n": 0})

    for r in all_results:
        section = r["section"]
        for key in ("tp", "fp", "fn", "tn"):
            totals["overall"][key] += r[key]
            totals[section][key] += r[key]
        totals["overall"]["exact_match"] += int(r["exact_match"])
        totals["overall"]["n"] += 1
        totals[section]["exact_match"] += int(r["exact_match"])
        totals[section]["n"] += 1

    metrics: Dict[str, Any] = {}
    for name, counts in totals.items():
        p, r, f1 = _prf(counts["tp"], counts["fp"], counts["fn"])
        accuracy = (
            (counts["tp"] + counts["tn"]) / counts["n"] if counts["n"] > 0 else 0.0
        )
        exact_acc = counts["exact_match"] / counts["n"] if counts["n"] > 0 else 0.0
        metrics[name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "exact_match_accuracy": round(exact_acc, 4),
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "tn": counts["tn"],
            "n_fields": counts["n"],
        }
    return metrics


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def print_report(metrics: Dict[str, Any], per_record: List[Dict]) -> None:
    sections_order = ["I", "N", "O", "overall"]

    print("\n" + "=" * 70)
    print(" MDS 3.0 Extraction Evaluation Report")
    print("=" * 70)
    print(f"{'Section':<12} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Exact Acc':>10} {'N fields':>10}")
    print("-" * 70)
    for sec in sections_order:
        if sec not in metrics:
            continue
        m = metrics[sec]
        label = f"Section {sec}" if sec != "overall" else "OVERALL"
        print(
            f"{label:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>8.4f} {m['exact_match_accuracy']:>10.4f} {m['n_fields']:>10}"
        )
    print("=" * 70)

    print("\nPer-record summary:")
    print(f"  {'note_id':<24} {'matched':>8} {'total GT':>10} {'exact':>8}")
    print("  " + "-" * 52)
    for rec in per_record:
        results = rec["results"]
        n_gt = len([r for r in results if r["field_id"] in rec["gt_fields"]])
        n_exact = sum(1 for r in results if r["exact_match"] and r["field_id"] in rec["gt_fields"])
        print(f"  {rec['note_id']:<24} {len(results):>8} {n_gt:>10} {n_exact:>8}")

    # Per-field breakdown for mismatches
    all_mismatches = [
        r
        for rec in per_record
        for r in rec["results"]
        if not r["exact_match"]
    ]
    if all_mismatches:
        print(f"\nMismatched fields ({len(all_mismatches)} total):")
        print(f"  {'field_id':<12} {'GT value':<20} {'predicted':<20} {'note_id'}")
        print("  " + "-" * 72)
        for rec in per_record:
            for r in rec["results"]:
                if not r["exact_match"]:
                    gt_str = str(r["ground_truth"])[:18]
                    pred_str = str(r["predicted"])[:18]
                    print(f"  {r['field_id']:<12} {gt_str:<20} {pred_str:<20} {rec['note_id']}")


def write_csv_report(
    all_results: List[FieldResult],
    per_record: List[Dict],
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "note_id",
                "field_id",
                "section",
                "ground_truth",
                "predicted",
                "exact_match",
                "tp",
                "fp",
                "fn",
                "tn",
            ],
        )
        writer.writeheader()
        for rec in per_record:
            for r in rec["results"]:
                writer.writerow({**r, "note_id": rec["note_id"]})
    print(f"\nCSV report written to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MDS 3.0 extraction predictions against ground truth."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to the model output JSON (mds_assessments.json or mds_ino_form_ready.json).",
    )
    parser.add_argument(
        "--ground-truth",
        default="data/test/ground_truth_labels.json",
        help="Path to the ground-truth labels JSON (default: data/test/ground_truth_labels.json).",
    )
    parser.add_argument(
        "--output",
        default="output/evaluation_report.csv",
        help="Path for the CSV evaluation report (default: output/evaluation_report.csv).",
    )
    parser.add_argument(
        "--fields-key",
        default=None,
        help=(
            "Key in each prediction record that contains the field dict. "
            "Auto-detected from 'fields' or 'codes' if not set."
        ),
    )
    return parser.parse_args()


# Keys that appear in a flat extraction record but are NOT MDS item IDs.
_FLAT_RECORD_METADATA_KEYS = frozenset(
    {"note_id", "subject_id", "hadm_id", "confidence", "description",
     "confidence_notes", "metadata", "extractor", "sections"}
)


def get_fields(record: Dict, fields_key: Optional[str]) -> Dict[str, Any]:
    """
    Extract the MDS field dict from a prediction or ground-truth record.

    Handles two formats:
    1. Nested  – record has a "fields" / "codes" key: ``{"fields": {...}}``
    2. Flat    – MDS item IDs are top-level keys mixed with metadata keys
                 (the format returned by LLMExtractor and stored in
                 extraction_ground_truth.json).
    """
    # Explicit override
    if fields_key and fields_key in record:
        return record[fields_key]
    # Nested format
    for key in ("fields", "codes"):
        if key in record and isinstance(record[key], dict):
            return record[key]
    # Flat format: strip metadata keys, keep anything that looks like an MDS ID
    flat = {
        k: v
        for k, v in record.items()
        if k not in _FLAT_RECORD_METADATA_KEYS
    }
    return flat


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.predictions):
        sys.exit(f"ERROR: predictions file not found: {args.predictions}")
    if not os.path.isfile(args.ground_truth):
        sys.exit(f"ERROR: ground-truth file not found: {args.ground_truth}")

    gt_records = load_json_records(args.ground_truth)
    pred_records = load_json_records(args.predictions)

    gt_index = build_index(gt_records)
    pred_index = build_index(pred_records)

    matched_note_ids = set(gt_index) & set(pred_index)
    unmatched_gt = set(gt_index) - set(pred_index)
    unmatched_pred = set(pred_index) - set(gt_index)

    print(f"Ground truth records : {len(gt_index)}")
    print(f"Prediction records   : {len(pred_index)}")
    print(f"Matched by note_id   : {len(matched_note_ids)}")
    if unmatched_gt:
        print(f"  GT records with no prediction : {sorted(unmatched_gt)}")
    if unmatched_pred:
        print(f"  Predictions with no GT record : {sorted(unmatched_pred)}")

    per_record_results: List[Dict] = []
    all_field_results: List[FieldResult] = []

    for note_id in sorted(matched_note_ids):
        gt_rec = gt_index[note_id]
        pred_rec = pred_index[note_id]

        gt_fields = gt_rec.get("fields", {})
        pred_fields = get_fields(pred_rec, args.fields_key)

        results = evaluate_pair(pred_fields, gt_fields)
        per_record_results.append(
            {"note_id": note_id, "results": results, "gt_fields": set(gt_fields)}
        )
        all_field_results.extend(results)

    if not all_field_results:
        sys.exit("No matching records found — nothing to evaluate.")

    metrics = aggregate_metrics(all_field_results)
    print_report(metrics, per_record_results)
    write_csv_report(all_field_results, per_record_results, args.output)

    # Print summary JSON to stdout for easy programmatic consumption
    print("\nMetrics JSON:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
