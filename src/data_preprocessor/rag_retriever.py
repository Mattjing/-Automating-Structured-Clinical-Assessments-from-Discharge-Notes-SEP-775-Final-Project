"""
BioBERT-based semantic retriever for MDS evidence selection — Option 2.

Replaces the rule-based evidence scoring in the preprocessor with semantic
similarity search over BioBERT sentence embeddings. The retrieved context is
formatted as plain text and passed directly to the existing LLMExtractor
(GPT) — no architectural changes to the extractor or mapper are needed.

What is retained from the preprocessor
---------------------------------------
* ``clean_discharge_text``         — removes MIMIC placeholders and admin noise
* ``detect_assertion``             — NegEx tags (CONFIRMED / NEGATED / UNCERTAIN)
* ``format_structured_data_summary`` — structured ICD / Rx evidence block
* Conflict detection output        — ICD vs free-text authority signal

What is replaced
----------------
* Keyword-based paragraph scoring  → BioBERT cosine similarity ranking
* Full knowledge graph format      → flat labelled passage list

Recommended models
------------------
* ``"dmis-lab/biobert-base-cased-v1.2"``                              — BioBERT
* ``"microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"`` — BiomedBERT
* ``"NLP4Science/pubmedbert-full-text"``                               — PubMedBERT

Usage
-----
    from src.data_preprocessor.rag_retriever import BioBERTRetriever
    from src.extractor.extractor import LLMExtractor
    from src.mapper.mapper import MDSMapper
    from src.mds_schema import MDSSchema

    schema   = MDSSchema(section_ids=["I", "N", "O"])
    retriever = BioBERTRetriever(sections=["I", "N", "O"])
    extractor = LLMExtractor(schema=schema, api_key="sk-...", preprocess_input=False)
    mapper    = MDSMapper(schema=schema)

    # Build RAG context and pass it directly as the note text to the extractor
    context    = retriever.build_rag_context(note.text, note.metadata)
    raw        = extractor.extract(context)
    assessment = mapper.map(note.note_id, note.subject_id, note.hadm_id, raw)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.data_preprocessor.preprocessor import (
    clean_discharge_text,
    detect_assertion,
    format_structured_data_summary,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section query strings
# Each string represents what "section X evidence" looks and sounds like.
# BioBERT embeds these alongside candidate sentences and ranks by cosine sim.
# ---------------------------------------------------------------------------
_SECTION_QUERIES: Dict[str, str] = {
    "I": (
        "active disease diagnosis condition illness disorder "
        "hypertension diabetes heart failure stroke cancer infection "
        "atrial fibrillation chronic kidney disease COPD"
    ),
    "N": (
        "medication drug prescription dose route frequency "
        "tablet oral intravenous subcutaneous antibiotic "
        "warfarin insulin furosemide aspirin lisinopril metformin"
    ),
    "O": (
        "special treatment procedure therapy surgery dialysis "
        "blood transfusion mechanical ventilator physical therapy "
        "chemotherapy radiation PICC catheter BiPAP occupational therapy"
    ),
}


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def _resolve_device(prefer_gpu: bool, gpu_device: int) -> str:
    if not prefer_gpu:
        logger.info("BioBERTRetriever using CPU (GPU preference disabled).")
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        logger.warning("BioBERTRetriever using CPU (torch import failed: %s).", exc)
        return "cpu"
    if not torch.cuda.is_available():
        logger.info("BioBERTRetriever using CPU (CUDA not available).")
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count == 0:
        logger.info("BioBERTRetriever using CPU (no CUDA devices detected).")
        return "cpu"
    idx = gpu_device if gpu_device < device_count else 0
    if idx != gpu_device:
        logger.warning(
            "Requested CUDA device %d unavailable (device_count=%d). Falling back to 0.",
            gpu_device,
            device_count,
        )
    name = torch.cuda.get_device_name(idx)
    logger.info("BioBERTRetriever using GPU (CUDA device %d: %s).", idx, name)
    return f"cuda:{idx}"


# ---------------------------------------------------------------------------
# BioBERTRetriever
# ---------------------------------------------------------------------------
class BioBERTRetriever:
    """
    Semantic evidence retriever using BioBERT sentence embeddings.

    Splits the discharge note into sentences, embeds each sentence via
    BioBERT mean pooling, and selects the top-k most semantically similar
    sentences per MDS section using cosine similarity against pre-computed
    section query embeddings.

    Assertion detection (NegEx) is applied to every retrieved sentence so
    that GPT receives explicit CONFIRMED / NEGATED / UNCERTAIN labels —
    semantic retrieval alone cannot determine negation.

    Parameters
    ----------
    model_name:
        Hugging Face model ID for a BERT-style masked-LM model.
        Mean pooling over token embeddings is used to produce sentence vectors.
    sections:
        MDS section IDs to retrieve evidence for.
    prefer_gpu:
        Use CUDA when available.
    gpu_device:
        CUDA device index.
    top_k:
        Number of sentences to retrieve per section.
    min_sentence_tokens:
        Sentences shorter than this (whitespace-split token count) are skipped.
    """

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.2",
        sections: Optional[Sequence[str]] = None,
        prefer_gpu: bool = True,
        gpu_device: int = 0,
        top_k: int = 5,
        min_sentence_tokens: int = 4,
    ) -> None:
        self.sections = list(sections or ["I", "N", "O"])
        self.top_k = top_k
        self.min_sentence_tokens = min_sentence_tokens
        self.device = _resolve_device(prefer_gpu, gpu_device)

        logger.info("Loading BioBERT retriever model '%s'…", model_name)
        try:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._model.to(self.device)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load BioBERT model '{model_name}': {exc}"
            ) from exc

        # Pre-compute section query embeddings once at init time
        self._query_embeddings: Dict[str, Any] = {}
        for sec in self.sections:
            if sec in _SECTION_QUERIES:
                self._query_embeddings[sec] = self._embed([_SECTION_QUERIES[sec]])[0]

        logger.info("BioBERTRetriever ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_rag_context(
        self,
        note_text: str,
        note_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build a GPT-ready context string from semantically retrieved evidence.

        The returned string is a drop-in replacement for the context produced
        by ``build_extraction_context()`` and can be passed directly to
        ``LLMExtractor.extract()`` as the ``note_text`` argument.

        Parameters
        ----------
        note_text:
            Raw discharge note text.
        note_metadata:
            Optional dict — may contain ``structured_data`` attached by
            ``MIMICDischargeLoader``.

        Returns
        -------
        str
            Context block with retrieved passages (assertion-tagged),
            structured data summary, and section header.
        """
        cleaned = clean_discharge_text(note_text)
        sentences = self._split_sentences(cleaned)

        if not sentences:
            logger.warning("BioBERTRetriever: no sentences extracted from note.")
            return cleaned

        sentence_embeddings = self._embed(sentences)

        section_str = " ".join(self.sections)
        lines: List[str] = [
            f"=== TARGET SECTIONS ===\n{section_str}\n",
            "=== RETRIEVED EVIDENCE (BioBERT semantic retrieval) ===",
        ]

        seen: set = set()
        for section in self.sections:
            if section not in self._query_embeddings:
                continue
            top = self._retrieve_top_k(sentences, sentence_embeddings, section)
            for sent, score in top:
                if sent in seen:
                    continue
                seen.add(sent)
                assertion = detect_assertion(sent)
                lines.append(
                    f"[{section}][{assertion}][score={score:.3f}] {sent}"
                )

        # Structured data (ICD codes, Rx) — retained from preprocessor
        if note_metadata:
            structured = format_structured_data_summary(
                note_metadata, self.sections
            )
            if structured:
                lines.append("\n=== STRUCTURED EVIDENCE ===")
                lines.append(structured)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        """Split cleaned text into candidate sentences."""
        raw = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [
            s.strip()
            for s in raw
            if len(s.split()) >= self.min_sentence_tokens
        ]

    def _embed(self, texts: List[str]) -> List[Any]:
        """
        Embed a list of texts using BioBERT mean pooling.

        Returns a list of 1-D tensors, one per input text.
        """
        import torch

        if not texts:
            return []

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)

        # Mean pooling — average token embeddings weighted by attention mask
        token_embeddings = outputs.last_hidden_state          # (B, T, H)
        attention_mask   = encoded["attention_mask"]          # (B, T)
        mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask       = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings     = sum_embeddings / sum_mask            # (B, H)

        return [embeddings[i] for i in range(embeddings.size(0))]

    def _retrieve_top_k(
        self,
        sentences: List[str],
        embeddings: List[Any],
        section: str,
    ) -> List[Tuple[str, float]]:
        """
        Return the top-k (sentence, score) pairs most similar to section query.
        """
        import torch
        import torch.nn.functional as F

        query_emb  = self._query_embeddings[section].unsqueeze(0)  # (1, H)
        sent_matrix = torch.stack(embeddings)                       # (N, H)
        scores = F.cosine_similarity(query_emb, sent_matrix, dim=1) # (N,)

        k = min(self.top_k, len(sentences))
        top_indices = scores.topk(k).indices.tolist()
        return [(sentences[i], scores[i].item()) for i in top_indices]
