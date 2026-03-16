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
    assert "[i]" in joined
    assert "diagnosis" in joined
    assert "[n]" in joined
    assert "warfarin" in joined
    assert "[o]" in joined
    assert "oxygen therapy" in joined


def test_build_extraction_context_prioritizes_targeted_evidence():
    text = "Discharge diagnosis: COPD. Home medications reviewed."
    context = build_extraction_context(text, sections=["I", "N", "O"])
    assert "=== TARGET SECTIONS ===" in context
    assert "=== PRIORITY EVIDENCE ===" in context
    assert "copd" in context.lower()


def test_build_extraction_context_uses_supporting_excerpt_when_signal_is_sparse():
    text = "Short note without explicit section headers but mentions diabetes."
    context = build_extraction_context(text, sections=["I", "N", "O"])
    assert "=== SUPPORTING NOTE EXCERPT ===" in context
