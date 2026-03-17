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
from src.preprocessor import build_extraction_context

__version__ = "0.1.0"

__all__ = [
    "MIMICDischargeLoader",
    "MDSSchema",
    "MDSAssessment",
    "LLMExtractor",
    "MDSMapper",
    "ExtractionPipeline",
    "build_extraction_context",
]

"""
Automating Structured Clinical Assessments from Discharge Notes

A pipeline to extract MDS 3.0 form fields from MIMIC IV discharge summaries
using LLM-based NLP.
"""

from src.data_loader import MIMICDischargeLoader
from src.mds_schema import MDSSchema, MDSAssessment
from src.extractor import LLMExtractor
from src.mapper import MDSMapper
from src.medbert_extractor import MedBERTExtractor
from src.medbert_mapper import MedBERTMapper
from src.pipeline import ExtractionPipeline
from src.preprocessor import build_extraction_context

__version__ = "0.1.0"

__all__ = [
    "MIMICDischargeLoader",
    "MDSSchema",
    "MDSAssessment",
    "LLMExtractor",
    "MDSMapper",
    "MedBERTExtractor",
    "MedBERTMapper",
    "ExtractionPipeline",
    "build_extraction_context",
]
