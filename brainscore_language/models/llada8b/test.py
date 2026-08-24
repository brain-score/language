import pytest

from brainscore_language import load_model, model_registry


def test_registration():
    __import__("brainscore_language.models.llada8b")
    assert "llada-8b-base" in model_registry


@pytest.mark.memory_intense
def test_load_model():
    model = load_model("llada-8b-base")
    # Brain-Score's loader sets ``model.identifier`` to the registered string.
    assert model.identifier == "llada-8b-base"
