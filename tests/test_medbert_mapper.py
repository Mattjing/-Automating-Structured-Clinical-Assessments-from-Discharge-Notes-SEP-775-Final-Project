"""Tests for medbert_mapper.py."""

from src.mds_schema import MDSSchema
from src.mapper.medbert_mapper import MedBERTMapper


def test_medbert_mapper_attaches_metadata_and_maps_fields():
    schema = MDSSchema(section_ids=["I", "N", "O"])
    mapper = MedBERTMapper(schema=schema)

    extraction = {
        "I0700": True,
        "N0400E": 7,
        "confidence": {"I0700": 0.9, "N0400E": 0.85},
        "_evidence": {"I0700": ["history of hypertension"]},
        "_entities": [{"text": "hypertension", "label": "DISEASE", "score": 0.95}],
    }

    assessment = mapper.map("n1", "s1", "h1", extraction)

    assert assessment.get_field("I0700") is True
    assert assessment.get_field("N0400E") == 7
    assert assessment.metadata.get("extractor") == "medbert"
    assert "evidence" in assessment.metadata
    assert "entities" in assessment.metadata
