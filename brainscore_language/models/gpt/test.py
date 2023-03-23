import numpy as np
import pytest

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, expected_reading_times', [
    ('distilgpt2', [np.nan, 19.260605, 12.721411, 12.083241,
                    10.876629, 3.678278, 2.102749, 11.961533]),
    ('gpt2-xl', [np.nan, 1.378484e+01, 6.686095e+00, 2.284407e-01,
                 7.538393e-01, 6.105860e-03, 2.644155e-02, 4.411311e-03]),
    ('gpt-neo-2.7B', [np.nan, 15.07522869,  3.6358602 ,  0.04999408,  1.42219079,
                      0.0399301 ,  0.02614061,  0.02547451]),
    ('gpt-neo-1.3B', [np.nan, 15.36009979,  5.54412651,  0.11744193,  0.60116327,
                      0.04266951,  0.08952015,  0.09213546])
])
def test_reading_times(model_identifier, expected_reading_times):
    model = load_model(model_identifier)
    text = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy']
    model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
    reading_times = model.digest_text(text)['behavior']
    np.testing.assert_allclose(reading_times, expected_reading_times, atol=0.0001)


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, expected_next_words', [
    ('distilgpt2', ['es', 'the', 'fox']),
    ('gpt2-xl', ['jumps', 'the', 'dog']),
    ('gpt-neo-2.7B', ['jumps', 'the', 'dog']),
    ('gpt-neo-1.3B', ['jumps', 'the', 'dog'])
])
def test_next_word(model_identifier, expected_next_words):
    model = load_model(model_identifier)
    text = ['the quick brown fox', 'jumps over', 'the lazy']
    model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
    next_word_predictions = model.digest_text(text)['behavior']
    np.testing.assert_array_equal(next_word_predictions, expected_next_words)


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, feature_size', [
    ('distilgpt2', 768),
    ('gpt2-xl', 1600),
    ('gpt-neo-1.3B', 2048),
    ('gpt-neo-2.7B', 2560)
])
def test_neural(model_identifier, feature_size):
    model = load_model(model_identifier)
    text = ['the quick brown fox', 'jumps over', 'the lazy dog']
    model.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                 recording_type=ArtificialSubject.RecordingType.fMRI)
    representations = model.digest_text(text)['neural']
    assert len(representations['presentation']) == 3
    np.testing.assert_array_equal(representations['stimulus'], text)
    assert len(representations['neuroid']) == feature_size

    
