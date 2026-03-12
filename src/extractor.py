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
from typing import Any, Dict, List, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.mds_schema import MDSItem, MDSItemType, MDSSchema

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
    "For text fields use a string."
)

_USER_PROMPT_TEMPLATE = """\
Extract MDS 3.0 form fields from the following discharge note.

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

        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, note_text: str) -> Dict[str, Any]:
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
        items = self._get_items_to_extract()
        if not items:
            return {"confidence": {}}

        combined: Dict[str, Any] = {}
        combined_confidence: Dict[str, Any] = {}

        # Process in batches to avoid exceeding context window
        for i in range(0, len(items), self.items_per_request):
            batch = items[i : i + self.items_per_request]
            result = self._extract_batch(note_text, batch)
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
        prompt = _USER_PROMPT_TEMPLATE.format(
            note_text=note_text,
            fields_spec=fields_spec,
        )
        raw_response = self._call_llm(prompt)
        return self._parse_response(raw_response, items)

    def _call_llm(self, user_message: str) -> str:
        """Send a message to the LLM and return the text response."""
        if self.provider == "openai":
            return _call_openai(
                client=self._client,
                model=self.model,
                system_prompt=_SYSTEM_PROMPT,
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

        # Keep only expected keys + confidence; coerce types where possible
        result: Dict[str, Any] = {}
        item_map = {item.item_id: item for item in items}

        for item in items:
            value = parsed.get(item.item_id)
            if value is not None:
                result[item.item_id] = _coerce_value(value, item)

        confidence_raw = parsed.get("confidence", {})
        if isinstance(confidence_raw, dict):
            result["confidence"] = {
                k: float(v)
                for k, v in confidence_raw.items()
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
                return [str(v) for v in value]
            return [str(value)]
        # TEXT
        return str(value)
    except (ValueError, TypeError):
        logger.debug(
            "Could not coerce value %r to type %s for item %s; returning as-is.",
            value,
            item.item_type,
            item.item_id,
        )
        return value
