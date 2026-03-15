"""Load discharge input data and print sample records.

Example:
    python scripts/preview_input_data.py --source data/discharge.csv/discharge.csv --sample-size 3
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# Allow running this script directly from the repository root.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.data_loader import MIMICDischargeLoader


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
        default="data/sample_input_preview.csv",
        help="CSV file path to save sampled records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    loader = MIMICDischargeLoader(source=args.source)
    notes = loader.load()

    print(f"Loaded notes: {len(notes)}")
    if not notes:
        print("No records found.")
        return 0

    sample_count = max(0, min(args.sample_size, len(notes)))
    print(f"Showing sample records: {sample_count}")
    print("-" * 80)

    sample_rows = []
    for idx, note in enumerate(notes[:sample_count], start=1):
        charttime = note.metadata.get("charttime", "")
        storetime = note.metadata.get("storetime", "")
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

    with open(args.save_path, "w", newline="", encoding="utf-8") as fh:
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

    print(f"Saved sample data: {args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
