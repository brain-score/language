import numpy as np
import pytest

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject


@pytest.mark.memory_intense
def test_load_model():
    """Model can be loaded from the registry without errors."""
    model = load_model('phi-2')
    assert model is not None


@pytest.mark.memory_intense
def test_identifier():
    model = load_model('phi-2')
    assert model.identifier() == 'microsoft/phi-2'


@pytest.mark.memory_intense
def test_neural():
    """Model produces neural representations with the expected shape."""
    model = load_model('phi-2')
    text = ['the quick brown fox', 'jumps over', 'the lazy dog']
    model.start_neural_recording(
        recording_target=ArtificialSubject.RecordingTarget.language_system,
        recording_type=ArtificialSubject.RecordingType.fMRI,
    )
    representations = model.digest_text(text)['neural']
    assert len(representations['presentation']) == 3
    np.testing.assert_array_equal(representations['stimulus'], text)
    assert len(representations['neuroid']) == 2560

