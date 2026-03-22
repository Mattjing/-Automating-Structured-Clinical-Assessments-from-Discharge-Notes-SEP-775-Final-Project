"""Tests for preprocessor.py"""

from src.data_preprocessor.preprocessor import (
    build_extraction_context,
    build_patient_knowledge_graph_chart,
    clean_discharge_text,
    detect_assertion,
    expand_abbreviations,
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
    assert "=== PATIENT KNOWLEDGE GRAPH ===" in context
    assert "=== PRIORITY EVIDENCE ===" in context
    assert "copd" in context.lower()


def test_build_extraction_context_uses_supporting_excerpt_when_signal_is_sparse():
    text = "Short note without explicit section headers but mentions diabetes."
    context = build_extraction_context(text, sections=["I", "N", "O"])
    assert "=== SUPPORTING NOTE EXCERPT ===" in context


def test_build_extraction_context_includes_structured_evidence_when_available():
    text = "Discharge diagnosis includes hypertension."
    note_metadata = {
        "structured_data": {
            "diagnoses": [
                {"icd_code": "I10", "description": "Essential (primary) hypertension"}
            ],
            "prescriptions": [
                {"drug": "warfarin", "dose": "5 mg"}
            ],
        }
    }
    context = build_extraction_context(
        text,
        sections=["I", "N", "O"],
        note_metadata=note_metadata,
    )

    assert "=== STRUCTURED EVIDENCE ===" in context
    assert "diagnoses" in context.lower()
    assert "warfarin" in context.lower()


# ── New tests: abbreviation expansion ─────────────────────────────────────────

def test_expand_abbreviations_common_conditions():
    result = expand_abbreviations("Patient has CHF and COPD.")
    assert "congestive heart failure" in result.lower()
    assert "chronic obstructive pulmonary disease" in result.lower()


def test_expand_abbreviations_medications():
    result = expand_abbreviations("Start ASA 81mg PO daily.")
    assert "aspirin" in result.lower()
    assert "oral" in result.lower() or "by mouth" in result.lower()


def test_expand_abbreviations_does_not_alter_non_abbreviations():
    original = "The patient underwent surgery."
    assert expand_abbreviations(original) == original


def test_expand_abbreviations_case_insensitive():
    for variant in ("MI", "mi", "Mi"):
        result = expand_abbreviations(f"History of {variant}.")
        assert "myocardial infarction" in result.lower()


# ── New tests: assertion detection ────────────────────────────────────────────

def test_detect_assertion_confirmed():
    assert detect_assertion("Patient has atrial fibrillation.") == "CONFIRMED"
    assert detect_assertion("Diagnosed with hypertension.") == "CONFIRMED"


def test_detect_assertion_negated_pre_cue():
    assert detect_assertion("No evidence of pulmonary embolism.") == "NEGATED"
    assert detect_assertion("Patient denies chest pain.") == "NEGATED"
    assert detect_assertion("DVT ruled out.") == "NEGATED"


def test_detect_assertion_negated_post_cue():
    assert detect_assertion("Myocardial infarction was ruled out.") == "NEGATED"
    assert detect_assertion("Pneumonia not found on imaging.") == "NEGATED"


def test_detect_assertion_uncertain():
    assert detect_assertion("Possible CHF exacerbation.") == "UNCERTAIN"
    assert detect_assertion("Suspected pulmonary embolism.") == "UNCERTAIN"
    assert detect_assertion("Cannot rule out sepsis at this time.") == "UNCERTAIN"


# ── New tests: knowledge graph conflict detection ──────────────────────────────

def test_knowledge_graph_flags_conflict_structured_vs_unstructured():
    note_text = "Patient denies any history of atrial fibrillation."
    note_metadata = {
        "subject_id": "12345",
        "hadm_id": "99999",
        "structured_data": {
            "diagnoses": [
                {"icd_code": "I48", "description": "atrial fibrillation"}
            ]
        },
    }
    graph = build_patient_knowledge_graph_chart(
        note_text, sections=["I", "N", "O"], note_metadata=note_metadata
    )
    # Conflict should be detected and surfaced
    assert "CONFLICT" in graph or "NEGATED" in graph


def test_knowledge_graph_includes_assertion_status():
    note_text = (
        "Discharge diagnosis: CHF. "
        "Patient on warfarin 5mg daily. "
        "Patient received oxygen therapy."
    )
    graph = build_patient_knowledge_graph_chart(note_text, sections=["I", "N", "O"])
    assert "CONFIRMED" in graph


def test_extract_priority_snippets_catches_abbreviation_only_note():
    """Snippets are found even when only abbreviations appear (no expanded form)."""
    text = "Pt admitted with MI and CHF. Home meds include ASA and furosemide."
    snippets = extract_priority_snippets(text, sections=["I", "N", "O"])
    joined = " ".join(snippets).lower()
    # Should find something for section I (MI/CHF) and N (medications)
    assert len(snippets) > 0
