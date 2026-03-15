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
        self.sections = sections or ["I", "N", "O"]
        self.items_per_request = items_per_request
        self.output_dir = output_dir
        self.output_format = output_format.lower()
        self.include_source_text = include_source_text
        self.batch_size = batch_size
        self.strict_validation = strict_validation
        self.preprocess_input = preprocess_input

        # Shared schema instance
        self._schema = MDSSchema()

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

        for note in tqdm(notes, desc="Extracting MDS fields"):
            try:
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

        if errors:
            self._save_errors(errors)

        return assessments

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

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
    )
    assessments = pipeline.run()
    logger.info("Pipeline complete. Generated %d assessments.", len(assessments))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
