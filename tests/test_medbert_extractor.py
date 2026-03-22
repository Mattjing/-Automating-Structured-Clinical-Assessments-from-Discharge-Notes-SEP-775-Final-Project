"""Tests for medbert_extractor.py."""

from src.mds_schema import MDSSchema
from src.medbert_extractor import MedBERTExtractor, _resolve_transformers_device


class DummyNERPipeline:
    def __call__(self, _text: str):
        return [
            {"word": "hypertension", "entity_group": "DISEASE", "score": 0.95, "start": 10, "end": 22},
            {"word": "diabetes mellitus", "entity_group": "DISEASE", "score": 0.93, "start": 30, "end": 47},
            {"word": "warfarin", "entity_group": "CHEMICAL", "score": 0.91, "start": 60, "end": 68},
            {"word": "oxygen therapy", "entity_group": "TREATMENT", "score": 0.9, "start": 80, "end": 94},
        ]


def test_medbert_extract_returns_confidence_and_metadata_keys():
    schema = MDSSchema(section_ids=["I", "N", "O"])
    extractor = MedBERTExtractor(schema=schema, ner_pipeline=DummyNERPipeline())

    note = (
        "PMH includes hypertension and diabetes mellitus. "
        "Patient received warfarin for 7 days and oxygen therapy in hospital."
    )
    result = extractor.extract(note)

    assert "confidence" in result
    assert "_evidence" in result
    assert "_entities" in result


def test_medbert_extract_maps_i_n_o_fields():
    schema = MDSSchema(section_ids=["I", "N", "O"])
    extractor = MedBERTExtractor(schema=schema, ner_pipeline=DummyNERPipeline())

    note = (
        "History of hypertension and diabetes mellitus. "
        "On anticoagulant warfarin for 7 days. "
        "Required oxygen therapy."
    )
    result = extractor.extract(note)

    assert result.get("I0700") is True
    assert result.get("I2900") is True
    assert result.get("N0415E") == ["1"]
    assert result.get("O0110C1") is True


class _FakeCUDA:
    def __init__(self, available: bool, count: int):
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_name(self, index: int) -> str:
        return f"Fake GPU {index}"


class _FakeTorch:
    def __init__(self, cuda: _FakeCUDA):
        self.cuda = cuda


def test_resolve_transformers_device_uses_requested_cuda_index():
    fake_torch = _FakeTorch(cuda=_FakeCUDA(available=True, count=2))
    device = _resolve_transformers_device(True, 1, torch_module=fake_torch)
    assert device == 1


def test_resolve_transformers_device_falls_back_to_zero_for_invalid_index():
    fake_torch = _FakeTorch(cuda=_FakeCUDA(available=True, count=1))
    device = _resolve_transformers_device(True, 9, torch_module=fake_torch)
    assert device == 0


def test_resolve_transformers_device_returns_cpu_when_gpu_disabled():
    fake_torch = _FakeTorch(cuda=_FakeCUDA(available=True, count=1))
    device = _resolve_transformers_device(False, 0, torch_module=fake_torch)
    assert device == -1
