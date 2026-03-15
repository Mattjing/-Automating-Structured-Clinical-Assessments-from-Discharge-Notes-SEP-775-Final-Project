"""
Tests for mds_schema.py
"""

import json

import pytest

from src.mds_schema import (
    MDSAssessment,
    MDSItem,
    MDSItemType,
    MDSResponseOption,
    MDSSchema,
    MDSSection,
)


# ---------------------------------------------------------------------------
# MDSItem
# ---------------------------------------------------------------------------


class TestMDSItem:
    def test_option_codes(self):
        item = MDSItem(
            "A0800",
            "Gender",
            MDSItemType.SELECT,
            [MDSResponseOption("1", "Male"), MDSResponseOption("2", "Female")],
        )
        assert item.option_codes() == ["1", "2"]

    def test_option_labels(self):
        item = MDSItem(
            "A0800",
            "Gender",
            MDSItemType.SELECT,
            [MDSResponseOption("1", "Male"), MDSResponseOption("2", "Female")],
        )
        assert item.option_labels() == ["Male", "Female"]

    def test_no_options(self):
        item = MDSItem("C0500", "BIMS Score", MDSItemType.INTEGER)
        assert item.option_codes() == []


# ---------------------------------------------------------------------------
# MDSSection
# ---------------------------------------------------------------------------


class TestMDSSection:
    def test_get_item_found(self):
        item = MDSItem("A0800", "Gender", MDSItemType.SELECT)
        section = MDSSection("A", "Identification", items=[item])
        found = section.get_item("A0800")
        assert found is item

    def test_get_item_not_found(self):
        section = MDSSection("A", "Identification")
        assert section.get_item("ZZZZ") is None


# ---------------------------------------------------------------------------
# MDSSchema
# ---------------------------------------------------------------------------


class TestMDSSchema:
    @pytest.fixture
    def schema(self):
        return MDSSchema()

    def test_all_expected_sections_present(self, schema):
        expected = {"A", "B", "C", "D", "E", "G", "H", "I", "J", "K", "M", "N", "O"}
        assert expected.issubset(set(schema.section_ids()))

    def test_can_limit_schema_to_requested_sections(self):
        schema = MDSSchema(section_ids=["I", "N", "O"])
        assert schema.section_ids() == ["I", "N", "O"]
        assert schema.get_section("A") is None
        assert schema.get_item("I0700") is not None
        assert schema.get_item("A0800") is None

    def test_get_section_returns_section(self, schema):
        sec = schema.get_section("A")
        assert sec is not None
        assert sec.section_id == "A"

    def test_get_section_case_insensitive(self, schema):
        assert schema.get_section("a") is not None

    def test_get_section_missing_returns_none(self, schema):
        assert schema.get_section("Z") is None

    def test_get_item_found_across_sections(self, schema):
        item = schema.get_item("I0700")
        assert item is not None
        assert "Hypertension" in item.label

    def test_get_item_not_found_returns_none(self, schema):
        assert schema.get_item("XXXX") is None

    def test_all_items_non_empty(self, schema):
        items = schema.all_items()
        assert len(items) > 0

    def test_all_sections_sorted(self, schema):
        ids = [s.section_id for s in schema.all_sections()]
        assert ids == sorted(ids)

    def test_to_dict_is_dict(self, schema):
        d = schema.to_dict()
        assert isinstance(d, dict)
        assert "A" in d

    def test_to_json_parseable(self, schema):
        js = schema.to_json()
        parsed = json.loads(js)
        assert "A" in parsed

    def test_section_a_has_gender_item(self, schema):
        sec = schema.get_section("A")
        item = sec.get_item("A0800")
        assert item is not None
        codes = item.option_codes()
        assert "1" in codes and "2" in codes

    def test_section_i_has_diabetes(self, schema):
        item = schema.get_item("I2000")
        assert item is not None
        assert "Diabetes" in item.label

    def test_section_j_pain_presence_codes(self, schema):
        item = schema.get_item("J0300")
        assert item is not None
        assert "0" in item.option_codes()

    def test_section_g_adl_codes(self, schema):
        item = schema.get_item("G0110A1")
        assert item is not None
        assert "0" in item.option_codes()

    def test_section_m_pressure_ulcers(self, schema):
        item = schema.get_item("M0210")
        assert item is not None
        assert item.item_type == MDSItemType.BOOLEAN

    def test_section_n_medications(self, schema):
        item = schema.get_item("N0400A")
        assert item is not None
        assert item.item_type == MDSItemType.INTEGER

    def test_section_o_oxygen_therapy(self, schema):
        item = schema.get_item("O0100C1")
        assert item is not None
        assert item.item_type == MDSItemType.BOOLEAN


# ---------------------------------------------------------------------------
# MDSAssessment
# ---------------------------------------------------------------------------


class TestMDSAssessment:
    def test_set_and_get_field(self):
        a = MDSAssessment(note_id="n1", subject_id="p1", hadm_id="h1")
        a.set_field("I0700", True, confidence=0.9)
        assert a.get_field("I0700") is True
        assert a.confidence["I0700"] == pytest.approx(0.9)

    def test_get_field_missing_returns_none(self):
        a = MDSAssessment(note_id="n1", subject_id="p1", hadm_id="h1")
        assert a.get_field("NONEXISTENT") is None

    def test_to_dict_contains_expected_keys(self):
        a = MDSAssessment(note_id="n1", subject_id="p1", hadm_id="h1")
        a.set_field("I0700", True)
        d = a.to_dict()
        assert d["note_id"] == "n1"
        assert "fields" in d
        assert d["fields"]["I0700"] is True

    def test_to_flat_dict_has_field_as_top_level_key(self):
        a = MDSAssessment(note_id="n1", subject_id="p1", hadm_id="h1")
        a.set_field("J0300", "1")
        flat = a.to_flat_dict()
        assert flat["J0300"] == "1"
        assert flat["note_id"] == "n1"

    def test_default_confidence_is_one(self):
        a = MDSAssessment(note_id="n1", subject_id="p1", hadm_id="h1")
        a.set_field("I0600", True)
        assert a.confidence["I0600"] == pytest.approx(1.0)
