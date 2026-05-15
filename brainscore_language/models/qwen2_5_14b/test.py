import numpy as np
import pytest

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject


@pytest.mark.memory_intense
def test_load_model():
    model = load_model('qwen2.5-14b')
    assert model is not None


@pytest.mark.memory_intense
def test_neural():
    model = load_model('qwen2.5-14b')
    text = ['the quick brown fox', 'jumps over', 'the lazy dog']
    model.start_neural_recording(
        recording_target=ArtificialSubject.RecordingTarget.language_system,
        recording_type=ArtificialSubject.RecordingType.fMRI,
    )
    representations = model.digest_text(text)['neural']
    assert len(representations['presentation']) == 3
    np.testing.assert_array_equal(representations['stimulus'], text)
    assert len(representations['neuroid']) == 5120


@pytest.mark.memory_intense
def test_next_word():
    model = load_model('qwen2.5-14b')
    text = ['the quick brown fox', 'jumps over', 'the lazy']
    model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
    next_words = model.digest_text(text)['behavior']
    assert len(next_words) == 3
    for word in next_words.values:
        assert isinstance(word, str)
        assert len(word.strip()) > 0
