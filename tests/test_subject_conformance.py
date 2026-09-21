"""Domain adapters implement Subject through the legacy compatibility base."""
from brainscore_core.model_interface import Subject, UnifiedModel
from brainscore_language.compat.unified_adapter import LanguageModelAdapter


def test_language_adapter_is_subject():
    assert issubclass(LanguageModelAdapter, Subject)
    assert issubclass(LanguageModelAdapter, UnifiedModel)
