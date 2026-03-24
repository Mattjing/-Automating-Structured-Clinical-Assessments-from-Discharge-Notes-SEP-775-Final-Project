"""
Generate ground-truth MDS 3.0 labeled samples from MIMIC-IV structured tables.

This script reads MIMIC-IV structured data files (diagnoses_icd, prescriptions,
procedures_icd) and maps them to MDS 3.0 item values using the ICD-to-MDS
mapping tables.  The output is a CSV file with one row per hospital admission,
containing both the ground-truth MDS fields and a reference back to the
discharge note for that admission.

Output columns
--------------
* ``subject_id``, ``hadm_id`` — patient and admission identifiers
* One column per MDS item (e.g. ``I0100``, ``N0415A``, ``O0110F1``) with
  value ``True`` if the item is triggered by at least one structured code
* ``triggered_items`` — JSON list of all triggered MDS item IDs
* ``source_codes`` — JSON dict mapping each triggered item to the codes/drugs
  that triggered it (for explainability / auditing)

Usage
-----
    python scripts/generate_labeled_samples.py \\
        --diagnoses data/diagnoses_icd.csv \\
        --prescriptions data/prescriptions.csv \\
        --procedures data/procedures_icd.csv \\
        --output data/labeled_samples.csv \\
        --sections I N O

    # Minimal (diagnoses only):
    python scripts/generate_labeled_samples.py \\
        --diagnoses data/diagnoses_icd.csv \\
        --output data/labeled_samples.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.icd_to_mds import (
    drug_to_mds_items,
    icd_to_mds_items,
    procedure_to_mds_items,
)
from src.mds_schema import MDSSchema

logger = logging.getLogger(__name__)


def _read_csv(path: str) -> pd.DataFrame:
    """Read CSV or CSV.GZ file."""
    if path.endswith(".gz"):
        return pd.read_csv(path, dtype=str, compression="gzip")
    return pd.read_csv(path, dtype=str)


def _process_diagnoses(
    path: str,
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Process diagnoses_icd.csv → {hadm_id: {mds_item: {icd_codes}}}.

    Expected columns: subject_id, hadm_id, icd_code, icd_version
    """
    df = _read_csv(path).fillna("")
    logger.info("Read %d diagnosis rows from %s", len(df), path)

    # Normalize column names (MIMIC-IV uses lowercase)
    df.columns = [c.lower().strip() for c in df.columns]

    results: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    for _, row in df.iterrows():
        hadm_id = str(row.get("hadm_id", "")).strip()
        icd_code = str(row.get("icd_code", "")).strip()
        icd_version = int(row.get("icd_version", 10) or 10)

        if not hadm_id or not icd_code:
            continue

        mds_items = icd_to_mds_items(icd_code, icd_version)
        for item_id in mds_items:
            results[hadm_id][item_id].add(f"ICD{icd_version}:{icd_code}")

    logger.info(
        "Mapped diagnoses for %d admissions → %d total item triggers",
        len(results),
        sum(len(v) for v in results.values()),
    )
    return results


def _process_prescriptions(
    path: str,
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Process prescriptions.csv → {hadm_id: {mds_item: {drug_names}}}.

    Expected columns: subject_id, hadm_id, drug (or drug_name_generic)
    """
    df = _read_csv(path).fillna("")
    logger.info("Read %d prescription rows from %s", len(df), path)
    df.columns = [c.lower().strip() for c in df.columns]

    # Try different column names for drug
    drug_col = None
    for candidate in ["drug", "drug_name_generic", "drug_name_poe", "medication"]:
        if candidate in df.columns:
            drug_col = candidate
            break
    if drug_col is None:
        logger.warning(
            "No drug column found in prescriptions. Available: %s",
            list(df.columns),
        )
        return {}

    results: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    for _, row in df.iterrows():
        hadm_id = str(row.get("hadm_id", "")).strip()
        drug_name = str(row.get(drug_col, "")).strip()

        if not hadm_id or not drug_name:
            continue

        mds_items = drug_to_mds_items(drug_name)
        for item_id in mds_items:
            results[hadm_id][item_id].add(drug_name[:80])

    logger.info(
        "Mapped prescriptions for %d admissions → %d total item triggers",
        len(results),
        sum(len(v) for v in results.values()),
    )
    return results


def _process_procedures(
    path: str,
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Process procedures_icd.csv → {hadm_id: {mds_item: {procedure_descriptions}}}.

    Expected columns: subject_id, hadm_id, icd_code, long_title (or icd_code only)
    """
    df = _read_csv(path).fillna("")
    logger.info("Read %d procedure rows from %s", len(df), path)
    df.columns = [c.lower().strip() for c in df.columns]

    # Procedure description column
    desc_col = None
    for candidate in ["long_title", "description", "icd_code"]:
        if candidate in df.columns:
            desc_col = candidate
            break
    if desc_col is None:
        logger.warning("No description column found in procedures.")
        return {}

    results: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    for _, row in df.iterrows():
        hadm_id = str(row.get("hadm_id", "")).strip()
        desc = str(row.get(desc_col, "")).strip()

        if not hadm_id or not desc:
            continue

        mds_items = procedure_to_mds_items(desc)
        for item_id in mds_items:
            results[hadm_id][item_id].add(desc[:100])

    logger.info(
        "Mapped procedures for %d admissions → %d total item triggers",
        len(results),
        sum(len(v) for v in results.values()),
    )
    return results


def _merge_results(
    *sources: Dict[str, Dict[str, Set[str]]],
) -> Dict[str, Dict[str, Set[str]]]:
    """Merge multiple source results into a single {hadm_id: {item: {evidence}}} dict."""
    merged: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for source in sources:
        for hadm_id, items in source.items():
            for item_id, evidence in items.items():
                merged[hadm_id][item_id].update(evidence)
    return merged


def generate_labeled_dataframe(
    merged: Dict[str, Dict[str, Set[str]]],
    section_ids: List[str],
) -> pd.DataFrame:
    """
    Convert merged results into a labeled DataFrame.

    Each row is one hospital admission. Columns are MDS item IDs with
    boolean values, plus metadata columns.
    """
    schema = MDSSchema(section_ids=section_ids)
    all_item_ids = [item.item_id for item in schema.all_items()]

    rows: List[Dict[str, Any]] = []
    for hadm_id, item_evidence in sorted(merged.items()):
        row: Dict[str, Any] = {"hadm_id": hadm_id}

        triggered = set(item_evidence.keys())
        for item_id in all_item_ids:
            row[item_id] = item_id in triggered

        row["triggered_items"] = json.dumps(sorted(triggered))
        row["source_codes"] = json.dumps(
            {k: sorted(v) for k, v in item_evidence.items()},
            ensure_ascii=False,
        )
        row["num_triggered"] = len(triggered)
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        "Generated labeled samples: %d admissions, %d unique items across all.",
        len(df),
        df[all_item_ids].sum().astype(bool).sum() if len(df) > 0 else 0,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ground-truth MDS labeled samples from MIMIC-IV structured tables."
    )
    parser.add_argument(
        "--diagnoses", type=str, default=None,
        help="Path to diagnoses_icd.csv(.gz)",
    )
    parser.add_argument(
        "--prescriptions", type=str, default=None,
        help="Path to prescriptions.csv(.gz)",
    )
    parser.add_argument(
        "--procedures", type=str, default=None,
        help="Path to procedures_icd.csv(.gz)",
    )
    parser.add_argument(
        "--output", type=str, default="data/labeled_samples.csv",
        help="Output CSV path (default: data/labeled_samples.csv)",
    )
    parser.add_argument(
        "--sections", nargs="+", default=["I", "N", "O"],
        help="MDS sections to generate labels for (default: I N O)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    sources: List[Dict[str, Dict[str, Set[str]]]] = []

    if args.diagnoses:
        sources.append(_process_diagnoses(args.diagnoses))
    if args.prescriptions:
        sources.append(_process_prescriptions(args.prescriptions))
    if args.procedures:
        sources.append(_process_procedures(args.procedures))

    if not sources:
        parser.error(
            "At least one of --diagnoses, --prescriptions, or --procedures "
            "must be provided."
        )

    merged = _merge_results(*sources)
    df = generate_labeled_dataframe(merged, args.sections)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved %d labeled samples to %s", len(df), args.output)

    # Print summary statistics
    item_cols = [c for c in df.columns if c.startswith(("I", "N", "O")) and c != "num_triggered"]
    if item_cols and len(df) > 0:
        prevalence = df[item_cols].mean().sort_values(ascending=False)
        print("\n=== Top 20 most prevalent MDS items ===")
        for item_id, pct in prevalence.head(20).items():
            print(f"  {item_id}: {pct:.1%} of admissions")
        print(f"\nAverage items triggered per admission: {df['num_triggered'].mean():.1f}")


if __name__ == "__main__":
    main()
