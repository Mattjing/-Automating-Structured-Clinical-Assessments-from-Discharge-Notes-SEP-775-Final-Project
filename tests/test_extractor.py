"""
Tests for extractor.py

The tests use mocks so that no real OpenAI API calls are made.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.extractor import LLMExtractor, _build_fields_spec, _coerce_value
from src.mds_schema import MDSItem, MDSItemType, MDSResponseOption, MDSSchema


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def schema():
    return MDSSchema()


@pytest.fixture
def extractor(schema):
    """Return an LLMExtractor with a dummy OpenAI client."""
    with patch("src.extractor._build_openai_client", return_value=MagicMock()):
        e = LLMExtractor(
            schema=schema,
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="sk-test",
            sections=["I", "N", "O"],
        )
    return e


SAMPLE_NOTE = (
    "Patient is a 78-year-old male admitted following a hip fracture. "
    "PMH: hypertension, type 2 diabetes mellitus, atrial fibrillation. "
    "Patient is on insulin and warfarin. "
    "Required oxygen therapy during hospitalization."
)

# A realistic JSON response from the LLM
SAMPLE_LLM_RESPONSE = json.dumps(
    {
        "I0300": True,
        "I0700": True,
        "I2000": True,
        "N0400E": 7,
        "N0350B": True,
        "O0100C1": True,
        "O0100G1": True,
        "confidence": {
            "I0300": 0.9,
            "I0700": 1.0,
            "I2000": 1.0,
            "N0400E": 0.9,
            "N0350B": 0.85,
            "O0100C1": 0.95,
            "O0100G1": 0.8,
        },
    }
)


# ---------------------------------------------------------------------------
# _build_fields_spec helper
# ---------------------------------------------------------------------------


class TestBuildFieldsSpec:
    def test_includes_item_id(self):
        item = MDSItem("A0800", "Gender", MDSItemType.SELECT, [MDSResponseOption("1", "Male")])
        spec = _build_fields_spec([item])
        assert "A0800" in spec

    def test_includes_options(self):
        item = MDSItem(
            "A0800",
            "Gender",
            MDSItemType.SELECT,
            [MDSResponseOption("1", "Male"), MDSResponseOption("2", "Female")],
        )
        spec = _build_fields_spec([item])
        assert "Male" in spec
        assert "Female" in spec

    def test_includes_description(self):
        item = MDSItem("C0500", "BIMS Score", MDSItemType.INTEGER, description="Range 0-15")
        spec = _build_fields_spec([item])
        assert "Range 0-15" in spec

    def test_empty_item_list(self):
        spec = _build_fields_spec([])
        assert spec == ""


# ---------------------------------------------------------------------------
# _coerce_value helper
# ---------------------------------------------------------------------------


class TestCoerceValue:
    def test_boolean_true_string(self):
        item = MDSItem("X", "X", MDSItemType.BOOLEAN)
        assert _coerce_value("true", item) is True

    def test_boolean_false_string(self):
        item = MDSItem("X", "X", MDSItemType.BOOLEAN)
        assert _coerce_value("no", item) is False

    def test_boolean_native(self):
        item = MDSItem("X", "X", MDSItemType.BOOLEAN)
        assert _coerce_value(True, item) is True

    def test_integer_string(self):
        item = MDSItem("X", "X", MDSItemType.INTEGER)
        assert _coerce_value("7", item) == 7

    def test_integer_float(self):
        item = MDSItem("X", "X", MDSItemType.INTEGER)
        assert _coerce_value(3.9, item) == 4

    def test_select_returns_string(self):
        item = MDSItem("X", "X", MDSItemType.SELECT)
        assert _coerce_value(1, item) == "1"

    def test_multi_list(self):
        item = MDSItem("X", "X", MDSItemType.MULTI)
        assert _coerce_value(["A", "B"], item) == ["A", "B"]

    def test_multi_single_string(self):
        item = MDSItem("X", "X", MDSItemType.MULTI)
        assert _coerce_value("A", item) == ["A"]

    def test_text(self):
        item = MDSItem("X", "X", MDSItemType.TEXT)
        assert _coerce_value(42, item) == "42"

    def test_none_returns_none(self):
        item = MDSItem("X", "X", MDSItemType.BOOLEAN)
        assert _coerce_value(None, item) is None


# ---------------------------------------------------------------------------
# LLMExtractor
# ---------------------------------------------------------------------------


class TestLLMExtractorInit:
    def test_unknown_provider_raises(self, schema):
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMExtractor(schema=schema, provider="unknown_llm")

    def test_section_filter_uppercase(self, schema):
        with patch("src.extractor._build_openai_client", return_value=MagicMock()):
            e = LLMExtractor(schema=schema, api_key="sk-test", sections=["a", "i"])
        assert "A" in e.sections
        assert "I" in e.sections


class TestLLMExtractorExtract:
    def _make_extractor_with_mock_response(self, schema, response_json: str):
        """Return an LLMExtractor whose LLM call always returns *response_json*."""
        with patch("src.extractor._build_openai_client", return_value=MagicMock()):
            e = LLMExtractor(
                schema=schema,
                api_key="sk-test",
                sections=["I", "N", "O"],
                items_per_request=50,
            )
        e._call_llm = MagicMock(return_value=response_json)
        return e

    def test_extract_returns_dict(self, schema):
        e = self._make_extractor_with_mock_response(schema, SAMPLE_LLM_RESPONSE)
        result = e.extract(SAMPLE_NOTE)
        assert isinstance(result, dict)

    def test_extract_contains_known_field(self, schema):
        e = self._make_extractor_with_mock_response(schema, SAMPLE_LLM_RESPONSE)
        result = e.extract(SAMPLE_NOTE)
        assert "I0700" in result
        assert result["I0700"] is True

    def test_extract_contains_n_and_o_fields(self, schema):
        e = self._make_extractor_with_mock_response(schema, SAMPLE_LLM_RESPONSE)
        result = e.extract(SAMPLE_NOTE)
        assert result["N0400E"] == 7
        assert result["O0100C1"] is True

    def test_extract_contains_confidence(self, schema):
        e = self._make_extractor_with_mock_response(schema, SAMPLE_LLM_RESPONSE)
        result = e.extract(SAMPLE_NOTE)
        assert "confidence" in result
        assert result["confidence"]["I0700"] == pytest.approx(1.0)

    def test_extract_handles_malformed_json(self, schema):
        e = self._make_extractor_with_mock_response(schema, "NOT JSON AT ALL")
        result = e.extract(SAMPLE_NOTE)
        # Should not raise; should return a dict (possibly empty) with confidence key
        assert isinstance(result, dict)
        assert "confidence" in result

    def test_extract_strips_markdown_fences(self, schema):
        fenced = "```json\n" + SAMPLE_LLM_RESPONSE + "\n```"
        e = self._make_extractor_with_mock_response(schema, fenced)
        result = e.extract(SAMPLE_NOTE)
        assert "I0700" in result

    def test_extract_batch_returns_list_of_same_length(self, schema):
        e = self._make_extractor_with_mock_response(schema, SAMPLE_LLM_RESPONSE)
        notes = [SAMPLE_NOTE, SAMPLE_NOTE, SAMPLE_NOTE]
        results = e.extract_batch(notes)
        assert len(results) == 3

    def test_extract_respects_section_filter(self, schema):
        """Items outside the requested sections should not appear in result."""
        with patch("src.extractor._build_openai_client", return_value=MagicMock()):
            e = LLMExtractor(
                schema=schema,
                api_key="sk-test",
                sections=["A"],  # Only section A
                items_per_request=50,
            )
        # Response includes section I items — they should not be in final result
        # because they are not in the items_to_extract list
        response = json.dumps({"A0800": "1", "I0700": True, "confidence": {"A0800": 0.9}})
        e._call_llm = MagicMock(return_value=response)
        result = e.extract(SAMPLE_NOTE)
        assert "A0800" in result
        assert "I0700" not in result  # outside requested section A

    def test_empty_note_does_not_raise(self, schema):
        e = self._make_extractor_with_mock_response(schema, json.dumps({"confidence": {}}))
        result = e.extract("")
        assert isinstance(result, dict)

    def test_extract_uses_preprocessing_context(self, schema):
        with patch("src.extractor._build_openai_client", return_value=MagicMock()):
            e = LLMExtractor(
                schema=schema,
                api_key="sk-test",
                sections=["I", "N", "O"],
                items_per_request=50,
                preprocess_input=True,
            )
        e._call_llm = MagicMock(return_value=json.dumps({"confidence": {}}))

        with patch("src.extractor.build_extraction_context", return_value="=== CLEANED NOTE ===\ntext") as mock_ctx:
            e.extract(SAMPLE_NOTE)
            mock_ctx.assert_called_once()
