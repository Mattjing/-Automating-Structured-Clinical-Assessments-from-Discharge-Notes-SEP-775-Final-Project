"""
Tests for data_loader.py
"""

import os
import tempfile

import pandas as pd
import pytest

from src.data_loader import DischargeNote, MIMICDischargeLoader


# ---------------------------------------------------------------------------
# DischargeNote
# ---------------------------------------------------------------------------


class TestDischargeNote:
    def test_repr_shows_note_id(self):
        note = DischargeNote(
            note_id="n1",
            subject_id="p1",
            hadm_id="h1",
            text="Patient presents with chest pain.",
        )
        assert "n1" in repr(note)

    def test_repr_handles_empty_text(self):
        note = DischargeNote(note_id="n2", subject_id="p2", hadm_id="h2", text="")
        assert "n2" in repr(note)

    def test_metadata_defaults_to_empty_dict(self):
        note = DischargeNote(note_id="n3", subject_id="p3", hadm_id="h3", text="x")
        assert note.metadata == {}


# ---------------------------------------------------------------------------
# MIMICDischargeLoader — file-based loading
# ---------------------------------------------------------------------------


class TestMIMICDischargeLoaderFromFile:
    """Test loading from an Excel file."""

    @pytest.fixture
    def sample_excel(self, tmp_path):
        """Create a temporary Excel file with sample discharge notes."""
        df = pd.DataFrame(
            {
                "note_id": ["n1", "n2", "n3"],
                "subject_id": ["p1", "p2", "p3"],
                "hadm_id": ["h1", "h2", "h3"],
                "text": [
                    "Patient has hypertension and diabetes.",
                    "Admitted for hip fracture surgery.",
                    "Dementia with behavioural symptoms.",
                ],
                "charttime": ["2023-01-01", "2023-01-02", "2023-01-03"],
            }
        )
        path = str(tmp_path / "notes.xlsx")
        df.to_excel(path, index=False)
        return path

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a temporary CSV file with sample discharge notes."""
        df = pd.DataFrame(
            {
                "note_id": ["c1"],
                "subject_id": ["p10"],
                "hadm_id": ["h10"],
                "text": ["Patient requires oxygen therapy."],
            }
        )
        path = str(tmp_path / "notes.csv")
        df.to_csv(path, index=False)
        return path

    def test_load_excel_returns_correct_count(self, sample_excel):
        loader = MIMICDischargeLoader(source=sample_excel)
        notes = loader.load()
        assert len(notes) == 3

    def test_load_excel_note_fields(self, sample_excel):
        loader = MIMICDischargeLoader(source=sample_excel)
        notes = loader.load()
        assert notes[0].note_id == "n1"
        assert notes[0].subject_id == "p1"
        assert notes[0].hadm_id == "h1"
        assert "hypertension" in notes[0].text

    def test_load_excel_extra_columns_become_metadata(self, sample_excel):
        loader = MIMICDischargeLoader(source=sample_excel)
        notes = loader.load()
        # 'charttime' should appear in metadata
        assert "charttime" in notes[0].metadata

    def test_load_csv(self, sample_csv):
        loader = MIMICDischargeLoader(source=sample_csv)
        notes = loader.load()
        assert len(notes) == 1
        assert "oxygen" in notes[0].text

    def test_to_dataframe_has_correct_shape(self, sample_excel):
        loader = MIMICDischargeLoader(source=sample_excel)
        df = loader.to_dataframe()
        assert df.shape[0] == 3
        assert "note_id" in df.columns
        assert "text" in df.columns

    def test_get_notes_caches(self, sample_excel):
        loader = MIMICDischargeLoader(source=sample_excel)
        first = loader.get_notes()
        second = loader.get_notes()
        assert first is second  # same list object

    def test_missing_column_raises_value_error(self, tmp_path):
        df = pd.DataFrame({"note_id": ["n1"], "text": ["hello"]})
        path = str(tmp_path / "bad.xlsx")
        df.to_excel(path, index=False)
        loader = MIMICDischargeLoader(source=path)
        with pytest.raises(ValueError, match="missing"):
            loader.load()

    def test_unsupported_file_format_raises(self, tmp_path):
        path = str(tmp_path / "notes.txt")
        with open(path, "w") as f:
            f.write("hello")
        loader = MIMICDischargeLoader(source=path)
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load()

    def test_nonexistent_file_raises(self):
        loader = MIMICDischargeLoader(source="/nonexistent/path/notes.xlsx")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_custom_column_names(self, tmp_path):
        df = pd.DataFrame(
            {
                "id": ["x1"],
                "pid": ["pp1"],
                "admit": ["aa1"],
                "notes": ["Some note text."],
            }
        )
        path = str(tmp_path / "custom.csv")
        df.to_csv(path, index=False)
        loader = MIMICDischargeLoader(
            source=path,
            note_id_col="id",
            subject_id_col="pid",
            hadm_id_col="admit",
            text_col="notes",
        )
        notes = loader.load()
        assert len(notes) == 1
        assert notes[0].note_id == "x1"
        assert "Some note text" in notes[0].text


# ---------------------------------------------------------------------------
# MIMICDischargeLoader — pyhealth source
# ---------------------------------------------------------------------------


class TestMIMICDischargeLoaderPyhealth:
    def test_pyhealth_without_root_raises(self):
        loader = MIMICDischargeLoader(source="pyhealth", mimic_root="")
        with pytest.raises((ImportError, ValueError)):
            loader.load()

    def test_unknown_source_raises(self):
        loader = MIMICDischargeLoader(source="not_a_file_or_pyhealth")
        with pytest.raises(FileNotFoundError):
            loader.load()
