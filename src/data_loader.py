"""
Data loader for MIMIC IV discharge notes.

Supports two loading modes:
1. From an Excel/CSV file containing discharge notes as free text.
2. From the MIMIC IV dataset via the PyHealth library.
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DischargeNote:
    """Represents a single discharge note record."""

    def __init__(
        self,
        note_id: str,
        subject_id: str,
        hadm_id: str,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        self.note_id = note_id
        self.subject_id = subject_id
        self.hadm_id = hadm_id
        self.text = text
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ") if self.text else ""
        return (
            f"DischargeNote(note_id={self.note_id!r}, "
            f"subject_id={self.subject_id!r}, "
            f"hadm_id={self.hadm_id!r}, "
            f"text_preview={preview!r})"
        )


class MIMICDischargeLoader:
    """
    Loads MIMIC IV discharge notes from an Excel/CSV file or via PyHealth.

    Parameters
    ----------
    source : str
        Either a file path to an Excel/CSV file containing discharge notes,
        or the string ``"pyhealth"`` to load via the PyHealth MIMIC IV dataset.
    mimic_root : str, optional
        Root directory of the MIMIC IV dataset.  Required when
        ``source="pyhealth"``.
    note_id_col : str
        Column name for the note identifier.
    subject_id_col : str
        Column name for the patient/subject identifier.
    hadm_id_col : str
        Column name for the hospital admission identifier.
    text_col : str
        Column name containing the free-text discharge note.
    structured_sources : dict[str, str], optional
        Optional mapping of structured dataset names to file paths. Each
        structured file is loaded and attached to notes via matching IDs.
    structured_join_priority : sequence of str
        Identifier columns used to join structured rows to notes. Columns are
        attempted in order and only the columns present in each structured
        file are used.
    """

    def __init__(
        self,
        source: str = "pyhealth",
        mimic_root: str = "",
        note_id_col: str = "note_id",
        subject_id_col: str = "subject_id",
        hadm_id_col: str = "hadm_id",
        text_col: str = "text",
        structured_sources: Optional[Dict[str, str]] = None,
        structured_join_priority: Optional[Sequence[str]] = None,
    ) -> None:
        self.source = source
        self.mimic_root = mimic_root
        self.note_id_col = note_id_col
        self.subject_id_col = subject_id_col
        self.hadm_id_col = hadm_id_col
        self.text_col = text_col
        self.structured_sources = structured_sources or {}
        self.structured_join_priority = tuple(
            structured_join_priority
            or (self.note_id_col, self.hadm_id_col, self.subject_id_col)
        )
        self._notes: List[DischargeNote] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> List[DischargeNote]:
        """Load discharge notes and return a list of :class:`DischargeNote` objects.

        Returns
        -------
        List[DischargeNote]
            All discharge notes that were successfully loaded.
        """
        if self.source == "pyhealth":
            self._notes = self._load_from_pyhealth()
        elif os.path.isfile(self.source):
            self._notes = self._load_from_file(self.source)
        else:
            raise FileNotFoundError(
                f"Source '{self.source}' is not a valid file path and is not "
                "'pyhealth'. Please provide either a path to an Excel/CSV file "
                "or pass source='pyhealth'."
            )

        if self.structured_sources:
            self._attach_structured_data(self._notes)

        logger.info("Loaded %d discharge notes.", len(self._notes))
        return self._notes

    def get_notes(self) -> List[DischargeNote]:
        """Return already-loaded notes (calls :meth:`load` if needed)."""
        if not self._notes:
            self.load()
        return self._notes

    def to_dataframe(self) -> pd.DataFrame:
        """Return loaded notes as a :class:`pandas.DataFrame`."""
        notes = self.get_notes()
        return pd.DataFrame(
            [
                {
                    "note_id": n.note_id,
                    "subject_id": n.subject_id,
                    "hadm_id": n.hadm_id,
                    "text": n.text,
                    **n.metadata,
                }
                for n in notes
            ]
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_tabular_file(self, file_path: str) -> pd.DataFrame:
        lower_path = file_path.lower()
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(file_path, dtype=str)
        if ext == ".csv" or lower_path.endswith(".csv.gz"):
            return pd.read_csv(file_path, dtype=str)
        raise ValueError(
            f"Unsupported file format for '{file_path}'. Use .xlsx, .xls, .csv, or .csv.gz."
        )

    def _id_value(self, note: DischargeNote, col_name: str) -> str:
        if col_name == self.note_id_col:
            return note.note_id
        if col_name == self.subject_id_col:
            return note.subject_id
        if col_name == self.hadm_id_col:
            return note.hadm_id
        return ""

    def _attach_structured_data(self, notes: List[DischargeNote]) -> None:
        for dataset_name, dataset_path in self.structured_sources.items():
            if not os.path.isfile(dataset_path):
                raise FileNotFoundError(
                    f"Structured data source '{dataset_name}' was not found: {dataset_path}"
                )

            df = self._read_tabular_file(dataset_path).fillna("")
            available_join_cols = [
                col for col in self.structured_join_priority if col in df.columns
            ]
            if not available_join_cols:
                logger.warning(
                    "Skipping structured source '%s': none of join columns %s were found.",
                    dataset_name,
                    list(self.structured_join_priority),
                )
                continue

            grouped_rows: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
            for _, row in df.iterrows():
                join_key = tuple(str(row[col]) for col in available_join_cols)
                grouped_rows[join_key].append(
                    {
                        col: str(row[col])
                        for col in df.columns
                        if col not in available_join_cols
                    }
                )

            attached_count = 0
            for note in notes:
                note_key = tuple(
                    self._id_value(note, col_name) for col_name in available_join_cols
                )
                matches = grouped_rows.get(note_key, [])
                if not matches:
                    continue

                note.metadata.setdefault("structured_data", {})
                note.metadata["structured_data"][dataset_name] = matches
                attached_count += 1

            logger.info(
                "Attached structured dataset '%s' to %d notes using join columns %s.",
                dataset_name,
                attached_count,
                available_join_cols,
            )

    def _load_from_file(self, file_path: str) -> List[DischargeNote]:
        """Load discharge notes from an Excel or CSV file."""
        df = self._read_tabular_file(file_path)

        # Validate required columns
        required = {
            self.note_id_col,
            self.subject_id_col,
            self.hadm_id_col,
            self.text_col,
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"The following required columns are missing from the file: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.fillna("")
        extra_cols = [
            c
            for c in df.columns
            if c
            not in {
                self.note_id_col,
                self.subject_id_col,
                self.hadm_id_col,
                self.text_col,
            }
        ]

        notes: List[DischargeNote] = []
        for _, row in df.iterrows():
            note = DischargeNote(
                note_id=str(row[self.note_id_col]),
                subject_id=str(row[self.subject_id_col]),
                hadm_id=str(row[self.hadm_id_col]),
                text=str(row[self.text_col]),
                metadata={col: row[col] for col in extra_cols},
            )
            notes.append(note)
        return notes

    def _load_from_pyhealth(self) -> List[DischargeNote]:
        """Load discharge notes from MIMIC IV using the PyHealth library."""
        try:
            from pyhealth.datasets import MIMIC4Dataset
        except ImportError as exc:
            raise ImportError(
                "PyHealth is required when source='pyhealth'. "
                "Install it with: pip install pyhealth"
            ) from exc

        if not self.mimic_root:
            raise ValueError(
                "mimic_root must be set to the MIMIC IV data directory "
                "when using source='pyhealth'."
            )

        logger.info("Loading MIMIC IV dataset from %s via PyHealth…", self.mimic_root)

        dataset = MIMIC4Dataset(
            root=self.mimic_root,
            tables=["discharge"],
            code_mapping={},
            dev=False,
        )

        notes: List[DischargeNote] = []
        for patient_id, patient in dataset.patients.items():
            for visit in patient.visits.values():
                for event in visit.get_event_list("discharge"):
                    note = DischargeNote(
                        note_id=str(getattr(event, "note_id", event.code)),
                        subject_id=str(patient_id),
                        hadm_id=str(visit.visit_id),
                        text=str(getattr(event, "text", "")),
                        metadata={
                            "charttime": str(getattr(event, "timestamp", "")),
                        },
                    )
                    notes.append(note)
        return notes
