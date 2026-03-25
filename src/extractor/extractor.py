"""
LLM-based extractor for MDS 3.0 fields from discharge notes.

The extractor sends a structured prompt to an LLM (OpenAI GPT by default)
asking it to identify the values of specific MDS items based on the clinical
text of a discharge summary.  It returns the raw structured response which the
:mod:`mapper` module then maps to :class:`~mds_schema.MDSAssessment` objects.

Supported providers
-------------------
* ``"openai"`` — OpenAI Chat Completions API (requires ``OPENAI_API_KEY``
  environment variable or explicit ``api_key`` constructor argument).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, cast

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.mds_schema import MDSItem, MDSItemType, MDSSchema
from src.data_preprocessor.preprocessor import (
    build_extraction_context,
    build_patient_knowledge_graph_chart,
    clean_discharge_text,
    format_structured_data_summary,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a clinical data expert specialising in nursing home assessments. "
    "Your task is to extract structured information from hospital discharge "
    "summaries and map it to MDS 3.0 (Minimum Data Set) form fields.  "
    "Always respond with valid JSON only — no extra prose, no markdown "
    "code fences.  If information for a field is not mentioned or cannot be "
    "inferred from the text, set its value to null.  "
    "For boolean fields use true/false.  "
    "For select fields use the code string exactly as provided.  "
    "For integer fields use a number.  "
    "For multi-select fields use a JSON array of code strings.  "
    "For text fields use a string. "
    "For MDS 3.0 sections I, N, and O: map to official item coding only, and "
    "do not invent diagnosis, medication, or treatment codes when evidence is absent."
)

_EVIDENCE_SYSTEM_PROMPT = (
    "You are extracting only the most relevant evidence for MDS 3.0 sections I, N, and O "
    "from a hospital discharge note. Return valid JSON only. Use the exact section keys provided. "
    "For each section, return an array of short verbatim or near-verbatim evidence snippets from the note. "
    "Do not code MDS items. Do not infer absent facts. If no evidence is present for a section, return an empty array."
)

_USER_PROMPT_TEMPLATE = """\
Extract MDS 3.0 form fields from the following preprocessed discharge context.

=== DISCHARGE NOTE ===
{note_text}

=== MDS FIELDS TO EXTRACT ===
{fields_spec}

Respond with a single JSON object whose keys are the MDS item IDs listed above
and whose values are the extracted values (or null if not determinable).
Also include a "confidence" key containing a nested object with the same item
IDs mapped to a float between 0.0 (uncertain) and 1.0 (certain).

Example response format:
{{
  "A0800": "1",
  "I0700": true,
  "J0300": "1",
  "I8000": "Hypothyroidism, GERD",
  "confidence": {{
    "A0800": 0.95,
    "I0700": 1.0,
    "J0300": 0.8,
    "I8000": 0.9
  }}
}}
"""

_EVIDENCE_USER_PROMPT_TEMPLATE = """\
Extract the most relevant evidence snippets for MDS sections {sections} from the note below.

Return a single JSON object with exactly these keys:
- "I": evidence for diagnoses / active conditions
- "N": evidence for medications / insulin / anticoagulants / antibiotics / injections
- "O": evidence for treatments / procedures / oxygen / IV / dialysis / transfusions / ventilator support

Rules:
- Each value must be an array of short strings.
- Keep each string concise.
- Prefer direct evidence from the note.
- If there is no evidence for a section, return [].

=== DISCHARGE NOTE ===
{note_text}
"""

_SUPPORTED_PREPROCESSING_MODES = ("heuristic", "llm_evidence", "knowledge_graph")

_GRAPH_SYSTEM_PROMPT = (
    "You are a clinical data expert specialising in nursing home assessments. "
    "Your task is to extract structured information from a patient knowledge graph "
    "derived from a hospital discharge summary, and map it to MDS 3.0 (Minimum Data Set) form fields. "
    "The knowledge graph format is:\n"
    "  NODES — each node is tagged FACT[section][assertion] where assertion is one of:\n"
    "    CONFIRMED  — the clinical concept is explicitly asserted as present.\n"
    "    NEGATED    — the concept was explicitly ruled out or denied.\n"
    "    UNCERTAIN  — the concept is hedged, suspected, or a differential.\n"
    "  EDGES — P0 (the patient node) is connected to each evidence node via a semantic relation.\n"
    "  CONFLICTS — pairs of nodes that contradict each other across sources, with a resolution hint.\n"
    "Coding rules:\n"
    "- Only treat CONFIRMED nodes as positive evidence for MDS coding.\n"
    "- NEGATED nodes mean the condition/medication/treatment was ruled out — do NOT code them as present.\n"
    "- UNCERTAIN nodes may appear in text fields but should lower the confidence score, not be coded as definite.\n"
    "- For CONFLICTS, follow the resolution hint (prefer STRUCTURED source over unstructured negation).\n"
    "- Always respond with valid JSON only — no extra prose, no markdown code fences.\n"
    "- If information for a field is absent from the graph, set its value to null.\n"
    "- For boolean fields use true/false.  "
    "For select fields use the code string exactly as provided.  "
    "For integer fields use a number.  "
    "For multi-select fields use a JSON array of code strings.  "
    "For text fields use a string.\n"
    "- For MDS 3.0 sections I, N, and O: map to official item coding only, and "
    "do not invent diagnosis, medication, or treatment codes when evidence is absent."
)

_GRAPH_USER_PROMPT_TEMPLATE = """\
Extract MDS 3.0 form fields from the following patient knowledge graph.

{note_text}

=== MDS FIELDS TO EXTRACT ===
{fields_spec}

Instructions:
- Use CONFIRMED nodes as the primary evidence for coding decisions.
- NEGATED nodes represent explicitly ruled-out concepts — do not code them as present.
- UNCERTAIN nodes indicate hedged or suspected findings — note them in text fields only and
  reflect lower certainty in the confidence score.
- For any CONFLICT pair, follow the resolution hint in the CONFLICTS section of the graph.
- Set a field to null if no CONFIRMED evidence supports it in the graph.

Respond with a single JSON object whose keys are the MDS item IDs listed above
and whose values are the extracted values (or null if not determinable).
Also include a "confidence" key containing a nested object with the same item
IDs mapped to a float between 0.0 (uncertain) and 1.0 (certain).

Example response format:
{{
  "A0800": "1",
  "I0700": true,
  "J0300": "1",
  "I8000": "Hypothyroidism, GERD",
  "confidence": {{
    "A0800": 0.95,
    "I0700": 1.0,
    "J0300": 0.8,
    "I8000": 0.9
  }}
}}
"""


def _build_fields_spec(items: List[MDSItem]) -> str:
    """Build a human-readable spec string listing items and their options."""
    lines: List[str] = []
    for item in items:
        line = f"- {item.item_id}: {item.label} [{item.item_type.value}]"
        if item.options:
            opts = ", ".join(f"{o.code}={o.label!r}" for o in item.options)
            line += f"\n    Options: {opts}"
        if item.description:
            line += f"\n    Description: {item.description}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM Extractor
# ---------------------------------------------------------------------------


class LLMExtractor:
    """
    Extract MDS 3.0 field values from free-text discharge notes using an LLM.

    Parameters
    ----------
    schema : MDSSchema
        The MDS schema to extract fields for.
    provider : str
        LLM provider.  Currently supports ``"openai"`` only.
    model : str
        Model name (e.g. ``"gpt-3.5-turbo"``).
    api_key : str, optional
        API key for the provider.  Falls back to the ``OPENAI_API_KEY``
        environment variable when not provided.
    temperature : float
        Sampling temperature.  Use ``0.0`` for deterministic outputs.
    max_tokens : int
        Maximum tokens in the LLM response.
    max_retries : int
        Number of retry attempts on transient API errors.
    sections : list of str, optional
        List of section IDs to extract (e.g. ``["A", "I", "J"]``).
        Extracts all sections when empty or ``None``.
    items_per_request : int
        Maximum number of MDS items per LLM call.  Batches requests to stay
        within model context limits.
    """

    def __init__(
        self,
        schema: Optional[MDSSchema] = None,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_retries: int = 3,
        sections: Optional[List[str]] = None,
        items_per_request: int = 30,
        preprocess_input: bool = True,
    ) -> None:
        self.schema = schema or MDSSchema()
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.sections = [s.upper() for s in sections] if sections else []
        self.items_per_request = items_per_request
        self.preprocess_input = preprocess_input

        # Tracks the preprocessing mode active during the current extraction call
        # so that _extract_batch and _call_llm can select the appropriate prompts.
        self._active_preprocessing_mode: str = "heuristic"

        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        note_text: str,
        note_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract MDS field values from a single discharge note.

        Parameters
        ----------
        note_text : str
            The free-text content of the discharge summary.

        Returns
        -------
        dict
            A dictionary with MDS item IDs as keys and their extracted values.
            Includes a ``"confidence"`` key with per-item confidence scores.
        """
        prepared_text = self._prepare_note_text(note_text, note_metadata=note_metadata)
        items = self._get_items_to_extract()
        return self._extract_from_prepared_text(prepared_text, items)

    def extract_with_preprocessing_variants(
        self,
        note_text: str,
        modes: Optional[Sequence[str]] = None,
        note_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run extraction with multiple preprocessing modes and return all results."""
        items = self._get_items_to_extract()
        if not items:
            return {}

        selected_modes = [self._normalize_preprocessing_mode(mode) for mode in (modes or _SUPPORTED_PREPROCESSING_MODES)]
        results: Dict[str, Dict[str, Any]] = {}

        for mode in selected_modes:
            prepared_text = self._prepare_note_text(
                note_text,
                preprocessing_mode=mode,
                note_metadata=note_metadata,
            )
            extraction = self._extract_from_prepared_text(prepared_text, items)
            results[mode] = {
                "prepared_text": prepared_text,
                "extraction": extraction,
            }

        return results

    def _extract_from_prepared_text(
        self,
        prepared_text: str,
        items: List[MDSItem],
    ) -> Dict[str, Any]:
        """Extract MDS values from already prepared text."""
        if not items:
            return {"confidence": {}}

        combined: Dict[str, Any] = {}
        combined_confidence: Dict[str, Any] = {}

        # Process in batches to avoid exceeding context window
        for i in range(0, len(items), self.items_per_request):
            batch = items[i : i + self.items_per_request]
            result = self._extract_batch(prepared_text, batch)
            confidence = result.pop("confidence", {})
            combined.update(result)
            combined_confidence.update(confidence)

        combined["confidence"] = combined_confidence
        return combined

    def extract_batch(self, notes: List[str]) -> List[Dict[str, Any]]:
        """
        Extract MDS field values for multiple discharge notes.

        Parameters
        ----------
        notes : list of str
            List of discharge note texts.

        Returns
        -------
        list of dict
            One extraction dict per note.
        """
        return [self.extract(note) for note in notes]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_items_to_extract(self) -> List[MDSItem]:
        """Return the list of MDS items to extract based on section filter."""
        if self.sections:
            items: List[MDSItem] = []
            for sec_id in self.sections:
                section = self.schema.get_section(sec_id)
                if section:
                    items.extend(section.items)
            return items
        return self.schema.all_items()

    def _prepare_note_text(
        self,
        note_text: str,
        preprocessing_mode: str = "heuristic",
        note_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Preprocess note text before extraction to improve signal quality."""
        if not self.preprocess_input:
            self._active_preprocessing_mode = "heuristic"
            return note_text
        normalized_mode = self._normalize_preprocessing_mode(preprocessing_mode)
        self._active_preprocessing_mode = normalized_mode
        sections = self.sections or ["I", "N", "O"]
        if normalized_mode == "heuristic":
            return build_extraction_context(
                note_text,
                sections=sections,
                note_metadata=note_metadata,
            )
        if normalized_mode == "llm_evidence":
            return self._build_llm_evidence_context(
                note_text,
                sections=sections,
                note_metadata=note_metadata,
            )
        if normalized_mode == "knowledge_graph":
            return self._build_knowledge_graph_context(
                note_text,
                sections=sections,
                note_metadata=note_metadata,
            )
        raise ValueError(f"Unsupported preprocessing mode: {preprocessing_mode!r}")

    @staticmethod
    def _normalize_preprocessing_mode(mode: str) -> str:
        normalized = mode.strip().lower()
        if normalized not in _SUPPORTED_PREPROCESSING_MODES:
            raise ValueError(
                f"Unsupported preprocessing mode: {mode!r}. "
                f"Supported modes: {', '.join(_SUPPORTED_PREPROCESSING_MODES)}"
            )
        return normalized

    def _build_llm_evidence_context(
        self,
        note_text: str,
        sections: Sequence[str],
        note_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Use the LLM to condense the note into compact I/N/O evidence snippets."""
        cleaned = clean_discharge_text(note_text)
        if not cleaned:
            return ""

        prompt = _EVIDENCE_USER_PROMPT_TEMPLATE.format(
            sections=", ".join(section.upper() for section in sections),
            note_text=cleaned[:12000],
        )
        raw_response = self._call_evidence_llm(prompt)
        evidence = self._parse_evidence_response(raw_response, sections)

        parts: List[str] = [
            "=== TARGET SECTIONS ===\n" + ", ".join(section.upper() for section in sections),
            "=== LLM EVIDENCE SUMMARY ===",
        ]

        found_evidence = False
        for section in sections:
            section_id = section.upper()
            snippets = evidence.get(section_id, [])
            if snippets:
                found_evidence = True
                lines = "\n".join(f"- {snippet}" for snippet in snippets)
                parts.append(f"[{section_id}]\n{lines}")

        if not found_evidence:
            return build_extraction_context(
                note_text,
                sections=sections,
                note_metadata=note_metadata,
            )

        structured_block = format_structured_data_summary(
            note_metadata=note_metadata,
            sections=sections,
        )
        if structured_block:
            parts.append("=== STRUCTURED EVIDENCE ===\n" + structured_block)

        parts.append("=== SUPPORTING NOTE EXCERPT ===\n" + cleaned[:1800])
        return "\n\n".join(parts)

    def _build_knowledge_graph_context(
        self,
        note_text: str,
        sections: Sequence[str],
        note_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build context from the patient knowledge graph.

        Delegates directly to :func:`build_patient_knowledge_graph_chart` so
        the extractor receives a structured graph of CONFIRMED / NEGATED /
        UNCERTAIN evidence nodes rather than raw note prose.  Falls back to
        heuristic context when the graph builder returns an empty string (e.g.
        the note is blank and no structured metadata is available).
        """
        graph = build_patient_knowledge_graph_chart(
            note_text=note_text,
            sections=sections,
            note_metadata=note_metadata,
        )
        if not graph:
            logger.warning(
                "Knowledge-graph builder returned empty output; "
                "falling back to heuristic context."
            )
            return build_extraction_context(
                note_text,
                sections=sections,
                note_metadata=note_metadata,
            )
        return graph

    def _call_evidence_llm(self, user_message: str) -> str:
        """Call the same LLM client for evidence condensation."""
        if self.provider == "openai":
            return _call_openai(
                client=self._client,
                model=self.model,
                system_prompt=_EVIDENCE_SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.0,
                max_tokens=min(self.max_tokens, 900),
                max_retries=self.max_retries,
            )
        raise ValueError(f"Unsupported provider: {self.provider!r}")

    def _parse_evidence_response(
        self,
        raw: str,
        sections: Sequence[str],
    ) -> Dict[str, List[str]]:
        """Parse the LLM evidence condensation response."""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "Evidence preprocessor returned non-JSON response; falling back to heuristic context. "
                "Raw response (first 500 chars): %s",
                text[:500],
            )
            return {section.upper(): [] for section in sections}

        if not isinstance(parsed, dict):
            return {section.upper(): [] for section in sections}
        parsed_dict = cast(Dict[str, Any], parsed)

        result: Dict[str, List[str]] = {}
        for section in sections:
            section_id = section.upper()
            raw_snippets = parsed_dict.get(section_id, [])
            if isinstance(raw_snippets, list):
                snippets_list = cast(List[Any], raw_snippets)
                result[section_id] = [
                    str(snippet).strip()
                    for snippet in snippets_list
                    if str(snippet).strip()
                ][:8]
            else:
                result[section_id] = []

        return result

    def _build_client(self) -> Any:
        """Instantiate and return the LLM client."""
        if self.provider == "openai":
            return _build_openai_client(self.api_key)
        raise ValueError(
            f"Unsupported provider: {self.provider!r}. "
            "Currently supported providers: 'openai'."
        )

    def _extract_batch(
        self, note_text: str, items: List[MDSItem]
    ) -> Dict[str, Any]:
        """Send a single LLM request for a batch of items and parse the result."""
        fields_spec = _build_fields_spec(items)
        template = (
            _GRAPH_USER_PROMPT_TEMPLATE
            if self._active_preprocessing_mode == "knowledge_graph"
            else _USER_PROMPT_TEMPLATE
        )
        prompt = template.format(
            note_text=note_text,
            fields_spec=fields_spec,
        )
        raw_response = self._call_llm(prompt)
        return self._parse_response(raw_response, items)

    def _call_llm(self, user_message: str) -> str:
        """Send a message to the LLM and return the text response."""
        system_prompt = (
            _GRAPH_SYSTEM_PROMPT
            if self._active_preprocessing_mode == "knowledge_graph"
            else _SYSTEM_PROMPT
        )
        if self.provider == "openai":
            return _call_openai(
                client=self._client,
                model=self.model,
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_retries=self.max_retries,
            )
        raise ValueError(f"Unsupported provider: {self.provider!r}")

    def _parse_response(
        self, raw: str, items: List[MDSItem]
    ) -> Dict[str, Any]:
        """
        Parse the LLM JSON response.

        Falls back gracefully if the response is malformed.
        """
        text = raw.strip()
        # Strip markdown code fences if the model adds them despite instructions
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "LLM returned non-JSON response; falling back to null values.  "
                "Raw response (first 500 chars): %s",
                text[:500],
            )
            parsed = {}

        if not isinstance(parsed, dict):
            logger.warning(
                "LLM response was not a JSON object; falling back to null values."
            )
            parsed = {}
        parsed_dict = cast(Dict[str, Any], parsed)

        # Keep only expected keys + confidence; coerce types where possible
        result: Dict[str, Any] = {}
        item_map = {item.item_id: item for item in items}

        for item in items:
            value = parsed_dict.get(item.item_id)
            if value is not None:
                result[item.item_id] = _coerce_value(value, item)

        confidence_raw = parsed_dict.get("confidence", {})
        if isinstance(confidence_raw, dict):
            confidence_dict = cast(Dict[str, Any], confidence_raw)
            result["confidence"] = {
                k: float(v)
                for k, v in confidence_dict.items()
                if k in item_map and isinstance(v, (int, float))
            }
        else:
            result["confidence"] = {}

        return result


# ---------------------------------------------------------------------------
# OpenAI-specific helpers
# ---------------------------------------------------------------------------


def _build_openai_client(api_key: str) -> Any:
    """Build and return an OpenAI client instance."""
    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required. "
            "Install it with: pip install openai"
        ) from exc
    return OpenAI(api_key=api_key or None)


def _call_openai(
    client: Any,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> str:
    """Call the OpenAI chat completions API with retry logic."""
    try:
        from openai import APIError, RateLimitError  # type: ignore[import]
    except ImportError:
        APIError = Exception
        RateLimitError = Exception

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(max_retries),
        reraise=True,
    )
    def _inner() -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    return _inner()


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def _coerce_value(value: Any, item: MDSItem) -> Any:
    """Coerce *value* to the expected type for *item*."""
    if value is None:
        return None
    try:
        if item.item_type == MDSItemType.BOOLEAN:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"true", "yes", "1"}
            return bool(value)
        if item.item_type == MDSItemType.INTEGER:
            return int(round(float(value)))
        if item.item_type == MDSItemType.DATE:
            return str(value)
        if item.item_type == MDSItemType.SELECT:
            return str(value)
        if item.item_type == MDSItemType.MULTI:
            if isinstance(value, list):
                values = cast(List[Any], value)
                return [str(v) for v in values]
            return [str(value)]
        # TEXT
        return str(value)
    except (ValueError, TypeError):
        fallback_value = cast(Any, value)
        logger.debug(
            "Could not coerce value %r to type %s for item %s; returning as-is.",
            repr(fallback_value),
            item.item_type,
            item.item_id,
        )
        return fallback_value
