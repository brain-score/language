import numpy as np
import pytest

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, expected_reading_times', [
    ('lm1b', [28.3291111, 17.69329643, 15.81945515, 15.53895187, 11.78036213,
              4.01630831, 1.68043554, 14.99522114]),
])
def test_reading_times(model_identifier, expected_reading_times):
    model = load_model(model_identifier)
    text = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy']
    model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
    reading_times = model.digest_text(text)['behavior']
    np.testing.assert_allclose(reading_times, expected_reading_times, atol=0.0001)


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, expected_next_words', [
    ('lm1b', ['has', 'scene', 'House', ',', 'and', 'of']),
])
def test_next_word(model_identifier, expected_next_words):
    model = load_model(model_identifier)
    text = ["the president", "arrived to the", "White", "House", "after a long", "day"]
    model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
    next_word_predictions = model.digest_text(text)['behavior']
    np.testing.assert_array_equal(next_word_predictions, expected_next_words)


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, feature_size', [
    ('lm1b', 1024),
])
def test_neural(model_identifier, feature_size):
    model = load_model(model_identifier)
    text = ['the', 'quick brown fox', 'jumps over', 'the lazy dog']
    model.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                 recording_type=ArtificialSubject.RecordingType.fMRI)
    representations = model.digest_text(text)['neural']
    assert len(representations['presentation']) == len(text)
    np.testing.assert_array_equal(representations['stimulus'], text)
    assert len(representations['neuroid']) == feature_size
