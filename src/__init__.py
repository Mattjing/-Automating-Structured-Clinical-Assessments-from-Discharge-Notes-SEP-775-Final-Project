"""
Automating Structured Clinical Assessments from Discharge Notes

A pipeline to extract MDS 3.0 form fields from MIMIC IV discharge summaries
using LLM-based NLP.
"""

from src.data_loader import MIMICDischargeLoader
from src.mds_schema import MDSSchema, MDSAssessment
from src.extractor import LLMExtractor
from src.mapper import MDSMapper
from src.pipeline import ExtractionPipeline

__version__ = "0.1.0"

__all__ = [
    "MIMICDischargeLoader",
    "MDSSchema",
    "MDSAssessment",
    "LLMExtractor",
    "MDSMapper",
    "ExtractionPipeline",
]
