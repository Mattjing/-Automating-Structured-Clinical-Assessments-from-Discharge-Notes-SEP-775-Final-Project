"""Load discharge input data and print sample records.

Example:
    python scripts/preview_input_data.py --source data/discharge.csv/discharge.csv --sample-size 3
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, cast

import pandas as pd

# Allow running this script directly from the repository root.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.data_preprocessor.data_loader import MIMICDischargeLoader


_FIELD_DESCRIPTIONS = {
    "note_id": "Unique identifier for the discharge note record.",
    "subject_id": "Unique patient identifier in the source dataset.",
    "hadm_id": "Hospital admission identifier associated with the note.",
    "text": "Full free-text discharge note used as pipeline input.",
    "charttime": "Clinical chart timestamp associated with the note or event.",
    "storetime": "Timestamp when the note was stored in the source system.",
    "text_preview": "Shortened preview of the discharge note text for quick inspection.",
}


def _compact_text(text: str, limit: int) -> str:
    one_line = " ".join((text or "").split())
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit].rstrip() + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load discharge notes and show sample input rows."
    )
    parser.add_argument(
        "--source",
        default="data/discharge.csv/discharge.csv",
        help="Path to discharge input file (csv/xlsx/xls).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of sample records to print.",
    )
    parser.add_argument(
        "--text-preview-length",
        type=int,
        default=180,
        help="Maximum characters to print from each note text.",
    )
    parser.add_argument(
        "--save-path",
        default="data/sample_input_preview.xlsx",
        help="Preview output path. Use .xlsx for a multi-sheet workbook or .csv for a preview-only CSV.",
    )
    return parser.parse_args()


def _describe_field(field_name: str) -> str:
    if field_name in _FIELD_DESCRIPTIONS:
        return _FIELD_DESCRIPTIONS[field_name]
    return "Additional metadata column preserved from the input data source."


def _build_field_description_rows(columns: list[str]) -> list[dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for field_name in columns:
        rows.append(
            {
                "field_name": field_name,
                "description": _describe_field(field_name),
            }
        )
    return rows


def _save_preview_csv(save_path: str, sample_rows: list[dict[str, str]]) -> None:
    with open(save_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "note_id",
                "subject_id",
                "hadm_id",
                "charttime",
                "storetime",
                "text_preview",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)


def _save_preview_workbook(
    save_path: str,
    sample_rows: list[dict[str, str]],
    field_description_rows: list[dict[str, str]],
) -> None:
    with pd.ExcelWriter(save_path) as writer:
        writer_obj = cast(Any, writer)
        pd.DataFrame(sample_rows).to_excel(writer_obj, sheet_name="Sample Preview", index=False)
        pd.DataFrame(field_description_rows).to_excel(
            writer_obj,
            sheet_name="Input Fields",
            index=False,
        )


def main() -> int:
    args = parse_args()

    loader = MIMICDischargeLoader(source=args.source)
    notes = loader.load()
    source_df = loader.to_dataframe()

    print(f"Loaded notes: {len(notes)}")
    if not notes:
        print("No records found.")
        return 0

    sample_count = max(0, min(args.sample_size, len(notes)))
    print(f"Showing sample records: {sample_count}")
    print("-" * 80)

    sample_rows: List[Dict[str, str]] = []
    for idx, note in enumerate(notes[:sample_count], start=1):
        metadata = cast(Dict[str, Any], note.metadata)
        charttime = str(metadata.get("charttime", ""))
        storetime = str(metadata.get("storetime", ""))
        preview = _compact_text(note.text, limit=max(20, args.text_preview_length))

        print(f"[{idx}] note_id={note.note_id}")
        print(f"    subject_id={note.subject_id}, hadm_id={note.hadm_id}")
        if charttime or storetime:
            print(f"    charttime={charttime}, storetime={storetime}")
        print(f"    text_preview={preview}")
        print("-" * 80)

        sample_rows.append(
            {
                "note_id": note.note_id,
                "subject_id": note.subject_id,
                "hadm_id": note.hadm_id,
                "charttime": charttime,
                "storetime": storetime,
                "text_preview": preview,
            }
        )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    field_description_rows = _build_field_description_rows(list(source_df.columns))

    ext = os.path.splitext(args.save_path)[1].lower()
    if ext == ".csv":
        _save_preview_csv(args.save_path, sample_rows)

        workbook_path = os.path.splitext(args.save_path)[0] + ".xlsx"
        _save_preview_workbook(workbook_path, sample_rows, field_description_rows)
        print(f"Saved sample data: {args.save_path}")
        print(f"Saved workbook with field descriptions: {workbook_path}")
        return 0

    _save_preview_workbook(args.save_path, sample_rows, field_description_rows)

    print(f"Saved sample data: {args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
