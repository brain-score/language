"""v1.5: the language adapter conforms to the Subject interface.

``UnifiedModel`` was renamed to ``Subject`` in v1.5 (with ``UnifiedModel`` kept
as a deprecated alias). The language adapter now subclasses ``Subject``; this
test confirms the conformance and that the alias identity still holds. Note the
legacy language base class ``ArtificialSubject`` is unrelated to the unified
``Subject`` ABC.
"""
from brainscore_core.model_interface import Subject, UnifiedModel
from brainscore_language.compat.unified_adapter import LanguageModelAdapter


def test_language_adapter_is_subject():
    assert issubclass(LanguageModelAdapter, Subject)
    assert issubclass(LanguageModelAdapter, UnifiedModel)  # alias identity
