"""
Automating Structured Clinical Assessments from Discharge Notes

A pipeline to extract MDS 3.0 form fields from MIMIC IV discharge summaries
using LLM-based NLP.
"""

from src.data_preprocessor.data_loader import MIMICDischargeLoader
from src.mds_schema import MDSSchema, MDSAssessment
from src.extractor.extractor import LLMExtractor
from src.extractor.seq2seq_extractor import Seq2SeqExtractor
from src.mapper.mapper import MDSMapper
from src.mapper.seq2seq_mapper import Seq2SeqMapper
from src.pipeline import ExtractionPipeline
from src.data_preprocessor.preprocessor import build_extraction_context
from src.data_preprocessor.seq2seq_preprocessor import build_seq2seq_input
from src.data_preprocessor.rag_retriever import BioBERTRetriever

__version__ = "0.1.0"

__all__ = [
    "MIMICDischargeLoader",
    "MDSSchema",
    "MDSAssessment",
    "LLMExtractor",
    "Seq2SeqExtractor",
    "MDSMapper",
    "Seq2SeqMapper",
    "ExtractionPipeline",
    "build_extraction_context",
    "build_seq2seq_input",
    "BioBERTRetriever",
]
