"""
Tests for mapper.py
"""

import pytest

from src.mapper import MDSMapper
from src.mds_schema import MDSAssessment, MDSSchema


@pytest.fixture
def schema():
    return MDSSchema()


@pytest.fixture
def mapper(schema):
    return MDSMapper(schema=schema, strict=False)


@pytest.fixture
def strict_mapper(schema):
    return MDSMapper(schema=schema, strict=True)


# ---------------------------------------------------------------------------
# Basic mapping
# ---------------------------------------------------------------------------


class TestMDSMapperBasic:
    def test_map_returns_assessment(self, mapper):
        extraction = {
            "I0700": True,
            "confidence": {"I0700": 1.0},
        }
        a = mapper.map("n1", "p1", "h1", extraction)
        assert isinstance(a, MDSAssessment)

    def test_map_sets_ids(self, mapper):
        a = mapper.map("note1", "subj1", "adm1", {"confidence": {}})
        assert a.note_id == "note1"
        assert a.subject_id == "subj1"
        assert a.hadm_id == "adm1"

    def test_boolean_field_mapped(self, mapper):
        extraction = {"I0700": True, "confidence": {"I0700": 0.9}}
        a = mapper.map("n1", "p1", "h1", extraction)
        assert a.get_field("I0700") is True

    def test_integer_field_mapped(self, mapper):
        extraction = {"J0600A": 7, "confidence": {}}
        a = mapper.map("n1", "p1", "h1", extraction)
        assert a.get_field("J0600A") == 7

    def test_select_field_valid_code(self, mapper):
        extraction = {"A0800": "1", "confidence": {}}
        a = mapper.map("n1", "p1", "h1", extraction)
        assert a.get_field("A0800") == "1"

    def test_multi_field_valid_codes(self, mapper):
        extraction = {"J1100": ["A", "B"], "confidence": {}}
        a = mapper.map("n1", "p1", "h1", extraction)
        assert a.get_field("J1100") == ["A", "B"]

    def test_text_field_mapped(self, mapper):
        extraction = {"I8000": "Hypothyroidism", "confidence": {}}
        a = mapper.map("n1", "p1", "h1", extraction)
        assert a.get_field("I8000") == "Hypothyroidism"

    def test_confidence_stored(self, mapper):
        extraction = {"I0700": True, "confidence": {"I0700": 0.85}}
        a = mapper.map("n1", "p1", "h1", extraction)
        assert a.confidence["I0700"] == pytest.approx(0.85)

    def test_null_value_skipped(self, mapper):
        extraction = {"I0700": None, "confidence": {}}
        a = mapper.map("n1", "p1", "h1", extraction)
        assert a.get_field("I0700") is None

    def test_unknown_item_id_skipped(self, mapper):
        extraction = {"UNKNOWN_XYZ": "value", "confidence": {}}
        a = mapper.map("n1", "p1", "h1", extraction)
        assert a.get_field("UNKNOWN_XYZ") is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestMDSMapperValidation:
    def test_invalid_select_code_discarded_in_lenient_mode(self, mapper):
        extraction = {"A0800": "99", "confidence": {}}  # 99 is not valid
        a = mapper.map("n1", "p1", "h1", extraction)
        assert a.get_field("A0800") is None

    def test_invalid_select_code_raises_in_strict_mode(self, strict_mapper):
        extraction = {"A0800": "99", "confidence": {}}
        with pytest.raises(ValueError):
            strict_mapper.map("n1", "p1", "h1", extraction)

    def test_invalid_integer_discarded_in_lenient_mode(self, mapper):
        extraction = {"J0600A": "not_a_number", "confidence": {}}
        a = mapper.map("n1", "p1", "h1", extraction)
        assert a.get_field("J0600A") is None

    def test_invalid_integer_raises_in_strict_mode(self, strict_mapper):
        extraction = {"J0600A": "not_a_number", "confidence": {}}
        with pytest.raises(ValueError):
            strict_mapper.map("n1", "p1", "h1", extraction)

    def test_invalid_multi_code_discarded(self, mapper):
        extraction = {"J1100": ["INVALID_CODE"], "confidence": {}}
        a = mapper.map("n1", "p1", "h1", extraction)
        assert a.get_field("J1100") is None

    def test_boolean_from_string_true(self, mapper):
        extraction = {"I0700": "yes", "confidence": {}}
        a = mapper.map("n1", "p1", "h1", extraction)
        assert a.get_field("I0700") is True

    def test_boolean_from_string_false(self, mapper):
        extraction = {"I0700": "false", "confidence": {}}
        a = mapper.map("n1", "p1", "h1", extraction)
        assert a.get_field("I0700") is False


# ---------------------------------------------------------------------------
# Batch mapping
# ---------------------------------------------------------------------------


class TestMDSMapperBatch:
    def test_map_batch_returns_list(self, mapper):
        records = [
            {"note_id": "n1", "subject_id": "p1", "hadm_id": "h1"},
            {"note_id": "n2", "subject_id": "p2", "hadm_id": "h2"},
        ]
        extractions = [
            {"I0700": True, "confidence": {"I0700": 1.0}},
            {"I0700": False, "confidence": {"I0700": 0.8}},
        ]
        results = mapper.map_batch(records, extractions)
        assert len(results) == 2

    def test_map_batch_preserves_ids(self, mapper):
        records = [{"note_id": "n1", "subject_id": "p1", "hadm_id": "h1"}]
        extractions = [{"confidence": {}}]
        results = mapper.map_batch(records, extractions)
        assert results[0].note_id == "n1"

    def test_map_batch_mismatched_lengths_raises(self, mapper):
        records = [{"note_id": "n1", "subject_id": "p1", "hadm_id": "h1"}]
        extractions = [{"confidence": {}}, {"confidence": {}}]
        with pytest.raises(ValueError, match="same length"):
            mapper.map_batch(records, extractions)

    def test_map_batch_does_not_mutate_extraction(self, mapper):
        """map_batch should not remove 'confidence' from the original dict."""
        records = [{"note_id": "n1", "subject_id": "p1", "hadm_id": "h1"}]
        extraction = {"I0700": True, "confidence": {"I0700": 1.0}}
        original_keys = set(extraction.keys())
        mapper.map_batch(records, [extraction])
        assert set(extraction.keys()) == original_keys
