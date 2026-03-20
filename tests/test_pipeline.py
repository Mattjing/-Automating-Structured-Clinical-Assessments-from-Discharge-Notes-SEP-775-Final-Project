"""
Tests for pipeline.py

The pipeline tests mock the LLM calls so no actual API requests are made.
"""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.pipeline import ExtractionPipeline, _prompt_for_processing_mode, _resolve_processing_mode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_excel(tmp_path):
    df = pd.DataFrame(
        {
            "note_id": ["n1", "n2"],
            "subject_id": ["p1", "p2"],
            "hadm_id": ["h1", "h2"],
            "text": [
                "Patient has hypertension and diabetes. Pain score 6/10.",
                "Admitted with hip fracture. On warfarin.",
            ],
        }
    )
    path = str(tmp_path / "notes.xlsx")
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def large_note_list():
    return [
        SimpleNamespace(
            note_id=f"n{i}",
            subject_id=f"p{i}",
            hadm_id=f"h{i}",
            text=f"Patient note {i} with hypertension and warfarin.",
        )
        for i in range(8)
    ]


MOCK_LLM_RESPONSE = json.dumps(
    {
        "I0700": True,
        "I2000": True,
        "N0400E": 7,
        "N0350B": True,
        "O0100C1": True,
        "O0100G1": True,
        "confidence": {
            "I0700": 1.0,
            "I2000": 1.0,
            "N0400E": 0.9,
            "N0350B": 0.85,
            "O0100C1": 0.95,
            "O0100G1": 0.8,
        },
    }
)


def _make_pipeline(sample_excel, tmp_path, output_format="json", **kwargs):
    """Helper that returns a fully mocked pipeline."""
    with patch("src.extractor._build_openai_client", return_value=MagicMock()):
        pipeline = ExtractionPipeline(
            source=sample_excel,
            openai_api_key="sk-test",
            output_dir=str(tmp_path / "output"),
            output_format=output_format,
            sections=["I", "N", "O"],
            **kwargs,
        )
    return pipeline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractionPipelineRun:
    def test_run_returns_assessments(self, sample_excel, tmp_path):
        pipeline = _make_pipeline(sample_excel, tmp_path)
        with patch.object(pipeline, "_schema"), \
             patch("src.extractor.LLMExtractor.extract", return_value=json.loads(MOCK_LLM_RESPONSE)):
            results = pipeline.run()
        assert len(results) == 2

    def test_run_creates_json_output(self, sample_excel, tmp_path):
        pipeline = _make_pipeline(sample_excel, tmp_path, output_format="json")
        with patch("src.extractor.LLMExtractor.extract", return_value=json.loads(MOCK_LLM_RESPONSE)):
            pipeline.run()
        output_path = os.path.join(str(tmp_path / "output"), "mds_assessments.json")
        assert os.path.isfile(output_path)

    def test_run_creates_form_ready_ino_json_output(self, sample_excel, tmp_path):
        pipeline = _make_pipeline(sample_excel, tmp_path, output_format="json")
        with patch("src.extractor.LLMExtractor.extract", return_value=json.loads(MOCK_LLM_RESPONSE)):
            pipeline.run()
        output_path = os.path.join(str(tmp_path / "output"), "mds_ino_form_ready.json")
        assert os.path.isfile(output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        assert len(rows) == 2
        assert rows[0]["sections"] == ["I", "N", "O"]
        assert "codes" in rows[0]

    def test_run_creates_csv_output(self, sample_excel, tmp_path):
        pipeline = _make_pipeline(sample_excel, tmp_path, output_format="csv")
        with patch("src.extractor.LLMExtractor.extract", return_value=json.loads(MOCK_LLM_RESPONSE)):
            pipeline.run()
        output_path = os.path.join(str(tmp_path / "output"), "mds_assessments.csv")
        assert os.path.isfile(output_path)

    def test_run_creates_excel_output(self, sample_excel, tmp_path):
        pipeline = _make_pipeline(sample_excel, tmp_path, output_format="excel")
        with patch("src.extractor.LLMExtractor.extract", return_value=json.loads(MOCK_LLM_RESPONSE)):
            pipeline.run()
        output_path = os.path.join(str(tmp_path / "output"), "mds_assessments.xlsx")
        assert os.path.isfile(output_path)

    def test_json_output_is_valid(self, sample_excel, tmp_path):
        pipeline = _make_pipeline(sample_excel, tmp_path)
        with patch("src.extractor.LLMExtractor.extract", return_value=json.loads(MOCK_LLM_RESPONSE)):
            pipeline.run()
        output_path = os.path.join(str(tmp_path / "output"), "mds_assessments.json")
        with open(output_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 2
        assert "fields" in data[0]

    def test_assessment_fields_populated(self, sample_excel, tmp_path):
        pipeline = _make_pipeline(sample_excel, tmp_path)
        with patch("src.extractor.LLMExtractor.extract", return_value=json.loads(MOCK_LLM_RESPONSE)):
            results = pipeline.run()
        assert results[0].get_field("I0700") is True
        assert results[0].get_field("N0400E") == 7
        assert results[0].get_field("O0100C1") is True

    def test_form_ready_json_contains_ino_codes_only(self, sample_excel, tmp_path):
        pipeline = _make_pipeline(sample_excel, tmp_path, output_format="json")
        with patch("src.extractor.LLMExtractor.extract", return_value=json.loads(MOCK_LLM_RESPONSE)):
            pipeline.run()

        output_path = os.path.join(str(tmp_path / "output"), "mds_ino_form_ready.json")
        with open(output_path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        codes = rows[0]["codes"]
        assert "I0700" in codes
        assert "N0400E" in codes
        assert "O0100C1" in codes
        assert all(k[0] in {"I", "N", "O"} for k in codes)

    def test_note_ids_preserved(self, sample_excel, tmp_path):
        pipeline = _make_pipeline(sample_excel, tmp_path)
        with patch("src.extractor.LLMExtractor.extract", return_value=json.loads(MOCK_LLM_RESPONSE)):
            results = pipeline.run()
        note_ids = {a.note_id for a in results}
        assert note_ids == {"n1", "n2"}

    def test_extraction_error_logged_but_continues(self, sample_excel, tmp_path):
        """If one note fails extraction, the others should still be processed."""
        pipeline = _make_pipeline(sample_excel, tmp_path)

        call_count = 0

        def mock_extract(text, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated API failure")
            return json.loads(MOCK_LLM_RESPONSE)

        with patch("src.extractor.LLMExtractor.extract", side_effect=mock_extract):
            results = pipeline.run()

        # One note failed, one succeeded
        assert len(results) == 1
        # Error log file should be written
        error_path = os.path.join(str(tmp_path / "output"), "extraction_errors.json")
        assert os.path.isfile(error_path)

    def test_empty_source_returns_empty_list(self, tmp_path):
        """Pipeline with an empty Excel file should return an empty list."""
        df = pd.DataFrame(
            {"note_id": [], "subject_id": [], "hadm_id": [], "text": []}
        )
        path = str(tmp_path / "empty.xlsx")
        df.to_excel(path, index=False)
        pipeline = _make_pipeline(path, tmp_path)
        with patch("src.extractor.LLMExtractor.extract", return_value=json.loads(MOCK_LLM_RESPONSE)):
            results = pipeline.run()
        assert results == []

    def test_preprocess_flag_is_passed_to_extractor(self, sample_excel, tmp_path):
        with patch("src.extractor._build_openai_client", return_value=MagicMock()):
            pipeline = ExtractionPipeline(
                source=sample_excel,
                openai_api_key="sk-test",
                output_dir=str(tmp_path / "output"),
                output_format="json",
                preprocess_input=False,
            )
        assert pipeline.preprocess_input is False

    def test_structured_sources_are_stored_on_pipeline(self, sample_excel, tmp_path):
        with patch("src.extractor._build_openai_client", return_value=MagicMock()):
            pipeline = ExtractionPipeline(
                source=sample_excel,
                openai_api_key="sk-test",
                output_dir=str(tmp_path / "output"),
                output_format="json",
                structured_sources={"diagnoses": "data/diagnoses.csv"},
            )
        assert "diagnoses" in pipeline.structured_sources

    def test_compare_preprocessing_methods_writes_comparison_file(self, sample_excel, tmp_path):
        pipeline = _make_pipeline(
            sample_excel,
            tmp_path,
            output_format="json",
            compare_preprocessing_methods=True,
        )
        comparison_payload = {
            "heuristic": {
                "prepared_text": "heuristic context",
                "extraction": json.loads(MOCK_LLM_RESPONSE),
            },
            "llm_evidence": {
                "prepared_text": "llm evidence context",
                "extraction": json.loads(MOCK_LLM_RESPONSE),
            },
        }

        with patch("src.extractor.LLMExtractor.extract_with_preprocessing_variants", return_value=comparison_payload):
            pipeline.run()

        output_path = os.path.join(
            str(tmp_path / "output"),
            "mds_ino_preprocessing_comparison.json",
        )
        assert os.path.isfile(output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        assert len(rows) == 2
        assert set(rows[0]["variants"].keys()) == {"heuristic", "llm_evidence"}
        assert rows[0]["variants"]["heuristic"]["prepared_text"] == "heuristic context"

        diff_output_path = os.path.join(
            str(tmp_path / "output"),
            "mds_ino_preprocessing_diff_summary.json",
        )
        assert os.path.isfile(diff_output_path)

        csv_output_path = os.path.join(
            str(tmp_path / "output"),
            "mds_ino_preprocessing_diff_summary.csv",
        )
        assert os.path.isfile(csv_output_path)

        with open(diff_output_path, "r", encoding="utf-8") as f:
            diff_rows = json.load(f)

        assert diff_rows == []

        csv_rows = pd.read_csv(csv_output_path)
        assert csv_rows.empty

    def test_diff_summary_contains_only_disagreements(self, sample_excel, tmp_path):
        pipeline = _make_pipeline(
            sample_excel,
            tmp_path,
            output_format="json",
            compare_preprocessing_methods=True,
        )

        heuristic_extraction = json.loads(MOCK_LLM_RESPONSE)
        llm_extraction = json.loads(MOCK_LLM_RESPONSE)
        llm_extraction["O0100C1"] = False
        llm_extraction["confidence"]["O0100C1"] = 0.2

        comparison_payload = {
            "heuristic": {
                "prepared_text": "heuristic context",
                "extraction": heuristic_extraction,
            },
            "llm_evidence": {
                "prepared_text": "llm evidence context",
                "extraction": llm_extraction,
            },
        }

        with patch("src.extractor.LLMExtractor.extract_with_preprocessing_variants", return_value=comparison_payload):
            pipeline.run()

        diff_output_path = os.path.join(
            str(tmp_path / "output"),
            "mds_ino_preprocessing_diff_summary.json",
        )
        with open(diff_output_path, "r", encoding="utf-8") as f:
            diff_rows = json.load(f)

        assert len(diff_rows) == 2
        assert diff_rows[0]["field_differences"] == [
            {"item_id": "O0100C1", "heuristic": True, "llm_evidence": False}
        ]
        assert diff_rows[0]["confidence_differences"] == [
            {"item_id": "O0100C1", "heuristic": 0.95, "llm_evidence": 0.2}
        ]

        csv_output_path = os.path.join(
            str(tmp_path / "output"),
            "mds_ino_preprocessing_diff_summary.csv",
        )
        csv_rows = pd.read_csv(csv_output_path)
        assert len(csv_rows) == 4
        assert set(csv_rows["difference_type"]) == {"field", "confidence"}
        assert set(csv_rows["item_id"]) == {"O0100C1"}

    def test_sample_first_mode_limits_notes_processed(self, sample_excel, tmp_path, large_note_list):
        pipeline = _make_pipeline(
            sample_excel,
            tmp_path,
            output_format="json",
            sample_size=3,
            process_all_notes=False,
        )

        with patch("src.pipeline.MIMICDischargeLoader.load", return_value=large_note_list), \
             patch("src.extractor.LLMExtractor.extract", return_value=json.loads(MOCK_LLM_RESPONSE)) as mock_extract:
            results = pipeline.run()

        assert len(results) == 3
        assert mock_extract.call_count == 3

    def test_process_all_notes_overrides_sample_limit(self, sample_excel, tmp_path, large_note_list):
        pipeline = _make_pipeline(
            sample_excel,
            tmp_path,
            output_format="json",
            sample_size=3,
            process_all_notes=True,
        )

        with patch("src.pipeline.MIMICDischargeLoader.load", return_value=large_note_list), \
             patch("src.extractor.LLMExtractor.extract", return_value=json.loads(MOCK_LLM_RESPONSE)) as mock_extract:
            results = pipeline.run()

        assert len(results) == len(large_note_list)
        assert mock_extract.call_count == len(large_note_list)


class TestPipelineCliModeSelection:
    def test_prompt_for_processing_mode_accepts_numeric_choice(self):
        with patch("builtins.input", return_value="2"):
            assert _prompt_for_processing_mode() == "sample-compare"

    def test_prompt_for_processing_mode_defaults_to_sample(self):
        with patch("builtins.input", return_value=""):
            assert _prompt_for_processing_mode() == "sample"

    def test_resolve_processing_mode_from_explicit_mode(self):
        args = SimpleNamespace(
            mode="full-compare",
            compare_preprocessing_methods=False,
            process_all_notes=False,
        )

        resolved = _resolve_processing_mode(args)
        assert resolved.compare_preprocessing_methods is True
        assert resolved.process_all_notes is True

    def test_resolve_processing_mode_prompts_in_interactive_terminal(self):
        args = SimpleNamespace(
            mode=None,
            compare_preprocessing_methods=False,
            process_all_notes=False,
        )

        with patch("src.pipeline.sys.stdin.isatty", return_value=True), \
             patch("src.pipeline._prompt_for_processing_mode", return_value="full"):
            resolved = _resolve_processing_mode(args)

        assert resolved.compare_preprocessing_methods is False
        assert resolved.process_all_notes is True
