"""Tests for preprocessor.py"""

from src.preprocessor import (
    build_extraction_context,
    clean_discharge_text,
    extract_priority_snippets,
)


def test_clean_discharge_text_removes_header_and_placeholders():
    raw = """
    Name: ___ Unit No: ___
    Admission Date: ___ Discharge Date: ___
    [**Known firstname**] has hypertension.
    """
    cleaned = clean_discharge_text(raw)
    assert "Name:" not in cleaned
    assert "Admission Date" not in cleaned
    assert "hypertension" in cleaned.lower()
    assert "[**" not in cleaned
    assert "___" not in cleaned


def test_extract_priority_snippets_for_ino_keywords():
    text = (
        "Discharge diagnosis includes CHF and diabetes. "
        "Discharge medications: insulin and warfarin. "
        "Patient required oxygen therapy during stay."
    )
    snippets = extract_priority_snippets(text, sections=["I", "N", "O"])
    joined = " ".join(snippets).lower()
    assert "diagnosis" in joined
    assert "medications" in joined
    assert "oxygen therapy" in joined


def test_build_extraction_context_includes_cleaned_note_label():
    text = "Discharge diagnosis: COPD. Home medications reviewed."
    context = build_extraction_context(text, sections=["I", "N", "O"])
    assert "=== CLEANED NOTE ===" in context
    assert "copd" in context.lower()
