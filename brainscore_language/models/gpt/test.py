import numpy as np
import pytest

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, expected_reading_times', [
    ('openai-gpt', [np.nan, 13.267056, 13.145847, 11.971989,
                    13.211679,  4.648034, 1.44646 , 15.464076]),
    ('distilgpt2', [np.nan, 19.260605, 12.721411, 12.083241,
                    10.876629, 3.678278, 2.102749, 11.961533]),
    ('gpt2', [np.nan, 13.006296, 12.126239,  9.79957 ,
                   8.603763, 3.442141, 1.312605, 10.238353]),
    ('gpt2-medium', [np.nan, 14.88489, 6.539810, 0.08106061,
                   0.6542016, 0.06957269, 4.027023e-03, 4.039307e-04]),
    ('gpt2-large', [np.nan, 13.776375, 5.054959, 0.620946,
                 0.522623, 0.102953, 0.038324, 0.021452]),
    ('gpt2-xl', [np.nan, 1.378484e+01, 6.686095e+00, 2.284407e-01,
                 7.538393e-01, 6.105860e-03, 2.644155e-02, 4.411311e-03]),
    ('gpt-neo-125m', [np.nan, 14.348133,  6.299568 ,  6.598476,  8.743038,
                      3.293406 ,  0.741776,  6.183576]),
    ('gpt-neo-2.7B', [np.nan, 15.07522869,  3.6358602 ,  0.04999408,  1.42219079,
                      0.0399301 ,  0.02614061,  0.02547451]),
    ('gpt-neo-1.3B', [np.nan, 15.36009979,  5.54412651,  0.11744193,  0.60116327,
                      0.04266951,  0.08952015,  0.09213546]),
    ('gpt2', [np.nan, 13.00629139, 12.12623215,  9.79956627,  8.60373306,
        3.44214535,  1.31260252, 10.23834896])
])
def test_reading_times(model_identifier, expected_reading_times):
    model = load_model(model_identifier)
    text = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy']
    model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
    reading_times = model.digest_text(text)['behavior']
    np.testing.assert_allclose(reading_times, expected_reading_times, atol=0.0001)


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, expected_next_words', [
    ('openai-gpt', ['.', 'the', 'dog']),
    ('distilgpt2', ['es', 'the', 'fox']),
    ('gpt2', ['es', 'the', ',']),
    ('gpt2-medium', ['jumps', 'the', 'dog']),
    ('gpt2-large', ['jumps', 'the', 'dog']),
    ('gpt2-xl', ['jumps', 'the', 'dog']),
    ('gpt-neo-125m', [',', 'the', 'dog']),
    ('gpt-neo-2.7B', ['jumps', 'the', 'dog']),
    ('gpt-neo-1.3B', ['jumps', 'the', 'dog']),
    ('gpt2', ['es', 'the', ','])
])
def test_next_word(model_identifier, expected_next_words):
    model = load_model(model_identifier)
    text = ['the quick brown fox', 'jumps over', 'the lazy']
    model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
    next_word_predictions = model.digest_text(text)['behavior']
    np.testing.assert_array_equal(next_word_predictions, expected_next_words)


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, feature_size', [
    ('openai-gpt', 768),
    ('distilgpt2', 768),
    ('gpt2', 768),
    ('gpt2-medium', 1024),
    ('gpt2-large', 1280),
    ('gpt2-xl', 1600),
    ('gpt-neo-125m', 768),
    ('gpt-neo-1.3B', 2048),
    ('gpt-neo-2.7B', 2560),
    ('gpt2', 768)
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
