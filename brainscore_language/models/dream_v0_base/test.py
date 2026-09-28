import pytest

from brainscore_language import load_model, model_registry


def test_registration():
    __import__("brainscore_language.models.dream_v0_base")
    assert "dream-v0-base-7b" in model_registry


@pytest.mark.memory_intense
def test_load_model():
    model = load_model("dream-v0-base-7b")
    assert model.identifier == "dream-v0-base-7b"
