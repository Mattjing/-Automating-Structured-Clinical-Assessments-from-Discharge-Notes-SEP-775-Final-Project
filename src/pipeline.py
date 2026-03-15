"""
End-to-end pipeline: load discharge notes → extract MDS fields → map → save.

Usage example
-------------
.. code-block:: python

    from src.pipeline import ExtractionPipeline

    pipeline = ExtractionPipeline(
        source="data/discharge_notes.xlsx",
        openai_api_key="sk-...",
        output_dir="output",
        output_format="json",
    )
    assessments = pipeline.run()
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.data_loader import MIMICDischargeLoader
from src.extractor import LLMExtractor
from src.mapper import MDSMapper
from src.mds_schema import MDSAssessment, MDSSchema

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Orchestrates the full extraction pipeline.

    Steps
    -----
    1. Load discharge notes via :class:`~data_loader.MIMICDischargeLoader`.
    2. For each note, call :meth:`~extractor.LLMExtractor.extract` to obtain
       raw MDS field values.
    3. Map the raw values to :class:`~mds_schema.MDSAssessment` objects via
       :class:`~mapper.MDSMapper`.
    4. Persist the assessments to the configured output directory.

    Parameters
    ----------
    source : str
        Passed directly to :class:`~data_loader.MIMICDischargeLoader`.
        Either a file path (Excel/CSV) or ``"pyhealth"``.
    mimic_root : str, optional
        Required when ``source="pyhealth"``.
    note_id_col, subject_id_col, hadm_id_col, text_col : str
        Column names for the file-based loader (ignored for pyhealth).
    provider : str
        LLM provider (``"openai"``).
    model : str
        LLM model name.
    openai_api_key : str, optional
        API key for OpenAI.  Falls back to ``OPENAI_API_KEY`` env var.
    temperature : float
        LLM sampling temperature.
    max_tokens : int
        Max tokens per LLM response.
    max_retries : int
        Retry attempts on transient API errors.
    sections : list of str, optional
        MDS sections to extract. Defaults to ``["I", "N", "O"]``.
    items_per_request : int
        Max MDS items per LLM call.
    output_dir : str
        Directory to write output files.
    output_format : str
        ``"json"``, ``"csv"``, or ``"excel"``.
    include_source_text : bool
        Whether to include the original discharge text in the output.
    batch_size : int
        Number of notes to process before flushing a progress log entry.
    strict_validation : bool
        When ``True``, raise on invalid extracted values.
    preprocess_input : bool
        Whether to clean and focus note text before extraction.
    compare_preprocessing_methods : bool
        When ``True``, run both heuristic and LLM-evidence preprocessing and
        save a side-by-side comparison artifact.
    sample_size : int
        Number of notes to process in the default sample-first mode.
    process_all_notes : bool
        When ``True``, process the entire loaded dataset instead of stopping
        after the initial sample.
    """

    def __init__(
        self,
        source: str = "pyhealth",
        mimic_root: str = "",
        note_id_col: str = "note_id",
        subject_id_col: str = "subject_id",
        hadm_id_col: str = "hadm_id",
        text_col: str = "text",
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        openai_api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_retries: int = 3,
        sections: Optional[List[str]] = None,
        items_per_request: int = 30,
        output_dir: str = "output",
        output_format: str = "json",
        include_source_text: bool = False,
        batch_size: int = 10,
        strict_validation: bool = False,
        preprocess_input: bool = True,
        compare_preprocessing_methods: bool = False,
        sample_size: int = 5,
        process_all_notes: bool = False,
    ) -> None:
        self.source = source
        self.mimic_root = mimic_root
        self.note_id_col = note_id_col
        self.subject_id_col = subject_id_col
        self.hadm_id_col = hadm_id_col
        self.text_col = text_col
        self.provider = provider
        self.model = model
        self.openai_api_key = openai_api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.sections = [section.upper() for section in sections] if sections else ["I", "N", "O"]
        self.items_per_request = items_per_request
        self.output_dir = output_dir
        self.output_format = output_format.lower()
        self.include_source_text = include_source_text
        self.batch_size = batch_size
        self.strict_validation = strict_validation
        self.preprocess_input = preprocess_input
        self.compare_preprocessing_methods = compare_preprocessing_methods
        self.sample_size = max(1, int(sample_size))
        self.process_all_notes = process_all_notes
        self._comparison_rows: List[Dict[str, Any]] = []

        # Shared schema instance
        self._schema = MDSSchema(section_ids=self.sections)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[MDSAssessment]:
        """
        Execute the full pipeline.

        Returns
        -------
        list of MDSAssessment
            All extracted and mapped assessments.
        """
        # 1. Load notes
        logger.info("Loading discharge notes from '%s'…", self.source)
        loader = MIMICDischargeLoader(
            source=self.source,
            mimic_root=self.mimic_root,
            note_id_col=self.note_id_col,
            subject_id_col=self.subject_id_col,
            hadm_id_col=self.hadm_id_col,
            text_col=self.text_col,
        )
        notes = loader.load()
        logger.info("Loaded %d notes.", len(notes))

        if not notes:
            logger.warning("No notes found — pipeline finished with empty output.")
            return []

        notes = self._select_notes_to_process(notes)

        # 2. Set up extractor and mapper
        extractor = LLMExtractor(
            schema=self._schema,
            provider=self.provider,
            model=self.model,
            api_key=self.openai_api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            sections=self.sections,
            items_per_request=self.items_per_request,
            preprocess_input=self.preprocess_input,
        )
        mapper = MDSMapper(schema=self._schema, strict=self.strict_validation)

        # 3. Extract and map
        assessments: List[MDSAssessment] = []
        errors: List[Dict[str, Any]] = []
        self._comparison_rows = []

        for note in tqdm(notes, desc="Extracting MDS fields"):
            try:
                if self.compare_preprocessing_methods and self.preprocess_input:
                    comparison_result = extractor.extract_with_preprocessing_variants(
                        note.text,
                        modes=["heuristic", "llm_evidence"],
                    )
                    comparison_assessments: Dict[str, MDSAssessment] = {}

                    for mode, payload in comparison_result.items():
                        comparison_assessments[mode] = mapper.map(
                            note_id=note.note_id,
                            subject_id=note.subject_id,
                            hadm_id=note.hadm_id,
                            extraction=payload["extraction"],
                        )

                    assessment = comparison_assessments["heuristic"]
                    if self.include_source_text:
                        assessment.metadata = {"source_text": note.text}
                    self._comparison_rows.append(
                        self._build_comparison_row(
                            note_id=note.note_id,
                            subject_id=note.subject_id,
                            hadm_id=note.hadm_id,
                            source_text=note.text if self.include_source_text else None,
                            comparison_result=comparison_result,
                            comparison_assessments=comparison_assessments,
                        )
                    )
                else:
                    raw = extractor.extract(note.text)
                    assessment = mapper.map(
                        note_id=note.note_id,
                        subject_id=note.subject_id,
                        hadm_id=note.hadm_id,
                        extraction=raw,
                    )
                    if self.include_source_text:
                        assessment.metadata = {"source_text": note.text}
                assessments.append(assessment)
            except Exception as exc:
                logger.error(
                    "Failed to process note %s: %s", note.note_id, exc, exc_info=True
                )
                errors.append({"note_id": note.note_id, "error": str(exc)})

        logger.info(
            "Extraction complete: %d succeeded, %d failed.",
            len(assessments),
            len(errors),
        )

        # 4. Save output
        if assessments:
            self._save(assessments)

        if self._comparison_rows:
            self._save_preprocessing_comparison(self._comparison_rows)
            self._save_preprocessing_diff_summary(self._comparison_rows)

        if errors:
            self._save_errors(errors)

        return assessments

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def _select_notes_to_process(self, notes: List[Any]) -> List[Any]:
        """Return either a sample subset or the full note list."""
        if self.process_all_notes:
            logger.info("Full dataset mode enabled. Processing all %d notes.", len(notes))
            return notes

        if len(notes) <= self.sample_size:
            logger.info(
                "Loaded note count (%d) is within sample size (%d); processing all notes.",
                len(notes),
                self.sample_size,
            )
            return notes

        logger.info(
            "Sample-first mode enabled. Processing the first %d of %d notes. "
            "Re-run with --process-all-notes to process the full dataset.",
            self.sample_size,
            len(notes),
        )
        return notes[: self.sample_size]

    def _save(self, assessments: List[MDSAssessment]) -> None:
        """Persist assessments to the configured output directory."""
        self._ensure_output_dir()

        if self.output_format == "json":
            self._save_json(assessments)
        elif self.output_format == "csv":
            self._save_csv(assessments)
        elif self.output_format in ("excel", "xlsx"):
            self._save_excel(assessments)
        else:
            logger.warning(
                "Unknown output format '%s'; defaulting to JSON.", self.output_format
            )
            self._save_json(assessments)

    def _save_json(self, assessments: List[MDSAssessment]) -> None:
        path = os.path.join(self.output_dir, "mds_assessments.json")
        data = [a.to_dict() for a in assessments]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        logger.info("Saved %d assessments to %s", len(assessments), path)

        form_ready_path = os.path.join(self.output_dir, "mds_ino_form_ready.json")
        form_ready_rows = [self._to_form_ready_codes(a) for a in assessments]
        with open(form_ready_path, "w", encoding="utf-8") as fh:
            json.dump(form_ready_rows, fh, indent=2, ensure_ascii=False)
        logger.info("Saved %d form-ready records to %s", len(assessments), form_ready_path)

    def _save_csv(self, assessments: List[MDSAssessment]) -> None:
        path = os.path.join(self.output_dir, "mds_assessments.csv")
        rows = [a.to_flat_dict() for a in assessments]
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info("Saved %d assessments to %s", len(assessments), path)

    def _save_excel(self, assessments: List[MDSAssessment]) -> None:
        path = os.path.join(self.output_dir, "mds_assessments.xlsx")
        rows = [a.to_flat_dict() for a in assessments]
        df = pd.DataFrame(rows)
        df.to_excel(path, index=False)
        logger.info("Saved %d assessments to %s", len(assessments), path)

    def _save_errors(self, errors: List[Dict[str, Any]]) -> None:
        self._ensure_output_dir()
        path = os.path.join(self.output_dir, "extraction_errors.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(errors, fh, indent=2)
        logger.warning("Saved %d extraction errors to %s", len(errors), path)

    def _save_preprocessing_comparison(self, rows: List[Dict[str, Any]]) -> None:
        self._ensure_output_dir()
        path = os.path.join(self.output_dir, "mds_ino_preprocessing_comparison.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)
        logger.info("Saved %d preprocessing comparison rows to %s", len(rows), path)

    def _save_preprocessing_diff_summary(self, rows: List[Dict[str, Any]]) -> None:
        """Save a compact file containing only heuristic vs LLM-evidence disagreements."""
        self._ensure_output_dir()
        diff_rows = self._build_preprocessing_diff_summary(rows)
        path = os.path.join(self.output_dir, "mds_ino_preprocessing_diff_summary.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(diff_rows, fh, indent=2, ensure_ascii=False)
        logger.info("Saved %d preprocessing diff summary rows to %s", len(diff_rows), path)

    def _build_preprocessing_diff_summary(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build a compact summary containing only note-level disagreements."""
        summary_rows: List[Dict[str, Any]] = []
        for row in rows:
            diff_row = self._extract_preprocessing_diff_row(row)
            if diff_row is not None:
                summary_rows.append(diff_row)
        return summary_rows

    def _extract_preprocessing_diff_row(
        self,
        row: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Return only the disagreements between heuristic and llm_evidence for one note."""
        variants = row.get("variants", {})
        heuristic = variants.get("heuristic")
        llm_evidence = variants.get("llm_evidence")
        if not isinstance(heuristic, dict) or not isinstance(llm_evidence, dict):
            return None

        heuristic_assessment = heuristic.get("assessment", {})
        llm_assessment = llm_evidence.get("assessment", {})
        if not isinstance(heuristic_assessment, dict) or not isinstance(llm_assessment, dict):
            return None

        heuristic_fields = heuristic_assessment.get("fields", {})
        llm_fields = llm_assessment.get("fields", {})
        heuristic_confidence = heuristic_assessment.get("confidence", {})
        llm_confidence = llm_assessment.get("confidence", {})
        if not isinstance(heuristic_fields, dict) or not isinstance(llm_fields, dict):
            return None
        if not isinstance(heuristic_confidence, dict) or not isinstance(llm_confidence, dict):
            return None

        field_differences: List[Dict[str, Any]] = []
        all_field_ids = sorted(set(heuristic_fields) | set(llm_fields))
        for item_id in all_field_ids:
            heuristic_value = heuristic_fields.get(item_id)
            llm_value = llm_fields.get(item_id)
            if heuristic_value != llm_value:
                field_differences.append(
                    {
                        "item_id": item_id,
                        "heuristic": heuristic_value,
                        "llm_evidence": llm_value,
                    }
                )

        confidence_differences: List[Dict[str, Any]] = []
        all_confidence_ids = sorted(set(heuristic_confidence) | set(llm_confidence))
        for item_id in all_confidence_ids:
            heuristic_value = heuristic_confidence.get(item_id)
            llm_value = llm_confidence.get(item_id)
            if heuristic_value != llm_value:
                confidence_differences.append(
                    {
                        "item_id": item_id,
                        "heuristic": heuristic_value,
                        "llm_evidence": llm_value,
                    }
                )

        if not field_differences and not confidence_differences:
            return None

        return {
            "note_id": row.get("note_id"),
            "subject_id": row.get("subject_id"),
            "hadm_id": row.get("hadm_id"),
            "sections": row.get("sections", self.sections),
            "field_differences": field_differences,
            "confidence_differences": confidence_differences,
        }

    def _build_comparison_row(
        self,
        note_id: str,
        subject_id: str,
        hadm_id: str,
        source_text: Optional[str],
        comparison_result: Dict[str, Dict[str, Any]],
        comparison_assessments: Dict[str, MDSAssessment],
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "note_id": note_id,
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "sections": self.sections,
            "variants": {},
        }
        if source_text is not None:
            row["source_text"] = source_text

        for mode, payload in comparison_result.items():
            assessment = comparison_assessments[mode]
            row["variants"][mode] = {
                "prepared_text": payload["prepared_text"],
                "raw_extraction": payload["extraction"],
                "assessment": assessment.to_dict(),
            }

        return row

    def _to_form_ready_codes(self, assessment: MDSAssessment) -> Dict[str, Any]:
        """Return a compact JSON row with only I/N/O coded item values."""
        fields = {
            item_id: value
            for item_id, value in assessment.fields.items()
            if item_id.startswith(("I", "N", "O"))
        }
        confidence = {
            item_id: score
            for item_id, score in assessment.confidence.items()
            if item_id in fields
        }
        return {
            "note_id": assessment.note_id,
            "subject_id": assessment.subject_id,
            "hadm_id": assessment.hadm_id,
            "sections": ["I", "N", "O"],
            "codes": fields,
            "confidence": confidence,
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MDS I/N/O extraction pipeline")
    parser.add_argument(
        "--source",
        default="data/discharge.csv/discharge.csv",
        help="Path to discharge notes CSV/XLSX or 'pyhealth'",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for generated output files",
    )
    parser.add_argument(
        "--output-format",
        default="json",
        choices=["json", "csv", "excel", "xlsx"],
        help="Output format",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="OpenAI model name",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (falls back to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=["I", "N", "O"],
        help="MDS sections to extract (default: I N O)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable input preprocessing before extraction",
    )
    parser.add_argument(
        "--compare-preprocessing-methods",
        action="store_true",
        help="Run both heuristic and LLM-evidence preprocessing and save comparison output",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of notes to process in default sample-first mode",
    )
    parser.add_argument(
        "--process-all-notes",
        action="store_true",
        help="Process the full input dataset instead of stopping after the sample run",
    )
    return parser


def _main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        # Continue even if python-dotenv is unavailable.
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _build_arg_parser().parse_args()

    pipeline = ExtractionPipeline(
        source=args.source,
        output_dir=args.output_dir,
        output_format=args.output_format,
        model=args.model,
        openai_api_key=args.api_key,
        sections=args.sections,
        preprocess_input=not args.no_preprocess,
        compare_preprocessing_methods=args.compare_preprocessing_methods,
        sample_size=args.sample_size,
        process_all_notes=args.process_all_notes,
    )
    assessments = pipeline.run()
    logger.info("Pipeline complete. Generated %d assessments.", len(assessments))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
