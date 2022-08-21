import numpy as np
import pytest

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, expected_reading_times', [
    ('distilgpt2', [6.1768091e+02, 3.5762270e+04, 3.1096322e+04, 1.7372783e+04,
                    1.8824682e+06, 7.3022788e+03, 6.1768091e+02, 5.1282520e+06]),
    ('gpt2-xl', [1.2396754e+04, 3.3883262e+04, 1.0283889e+04, 3.5502209e+02,
                 1.0453620e+08, 2.1390867e+03, 1.2396754e+04, 2.0912675e+08]),
])
def test_reading_times(model_identifier, expected_reading_times):
    model = load_model(model_identifier)
    text = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy']
    model.perform_behavioral_task(task=ArtificialSubject.Task.reading_times)
    reading_times = model.digest_text(text)['behavior']
    np.testing.assert_allclose(reading_times, expected_reading_times, atol=0.0001)


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, expected_next_words', [
    ('distilgpt2', ['es', ' the', ',']),
    ('gpt2-xl', [' jumps', ' the', ',']),
])
def test_next_word(model_identifier, expected_next_words):
    model = load_model(model_identifier)
    text = ['the quick brown fox', 'jumps over', 'the lazy']
    model.perform_behavioral_task(task=ArtificialSubject.Task.next_word)
    next_word_predictions = model.digest_text(text)['behavior']
    np.testing.assert_array_equal(next_word_predictions, expected_next_words)


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, feature_size', [
    ('distilgpt2', 768),
    ('gpt2-xl', 1600),
])
def test_neural(model_identifier, feature_size):
    model = load_model(model_identifier)
    text = ['the quick brown fox', 'jumps over', 'the lazy dog']
    model.perform_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                   recording_type=ArtificialSubject.RecordingType.fMRI)
    representations = model.digest_text(text)['neural']
    assert len(representations['presentation']) == 3
    np.testing.assert_array_equal(representations['context'], text)
    assert len(representations['neuroid']) == feature_size
