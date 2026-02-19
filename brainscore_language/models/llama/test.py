import numpy as np
import pytest

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, expected_reading_times', [
    ('llama-7b', [9.724511, 12.605466, 3.32503, 0.09871647, 0.725152, 0.04576033, 0.07947908, 0.08976307]),
    ('llama-13b',[10.53345, 11.900979, 2.576608, 0.09501585, 0.6747948, 0.06707504, 0.07982931, 0.13605802]),
    ('llama-33b', [11.483265, 12.449862, 1.7104287, 0.10519427, 0.9729844, 0.12699145, 0.23386568, 0.15289368]),
    ('alpaca-7b', [3.15336514e+01, 1.61361885e+01, 6.20819473e+00, 3.02336123e-02, 4.87159938e-01, 5.48269460e-03, 1.08295875e-02, 1.63752567e-02]),
    ('vicuna-7b', [1.4193897e+01, 1.4030097e+01, 4.5661983e+00, 1.7538711e-02, 5.8269405e-01, 3.2116382e-03, 8.8979863e-02, 7.2399867e-03]),
    ('vicuna-13b', [5.1001291e+00, 1.1878480e+01, 7.0294745e-02, 2.8342367e-03, 8.7360293e-03, 6.8028755e-03, 5.5397633e-02, 3.3574910e-03]),
    ('vicuna-33b', [4.8655987, 14.37647, 1.5682482, 0.02738321, 0.34660488, 0.04076412, 0.0271305, 0.03512227]),
])
def test_reading_times(model_identifier, expected_reading_times):
    model = load_model(model_identifier)
    text = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy']
    model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
    reading_times = model.digest_text(text)['behavior']
    np.testing.assert_allclose(reading_times, expected_reading_times, atol=0.01)


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, expected_next_words', [
    ('llama-7b', ['j', 'the', 'dog']),
    ('llama-13b', ['j', 'the', 'dog']),
    ('llama-33b', ['j', 'the', 'dog']),
    ('alpaca-7b', ['j', 'the', 'dog']),
    ('vicuna-7b', ['j', 'the', 'dog']),
    ('vicuna-13b', ['j', 'the', 'dog']),
    ('vicuna-33b', ['j', 'the', 'dog']),
])
def test_next_word(model_identifier, expected_next_words):
    model = load_model(model_identifier)
    text = ['the quick brown fox', 'jumps over', 'the lazy']
    model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
    next_word_predictions = model.digest_text(text)['behavior']
    np.testing.assert_array_equal(next_word_predictions, expected_next_words)


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, feature_size', [
    ('llama-7b', 4096),
    ('llama-13b', 5120),
    ('llama-33b', 6656),
    ('alpaca-7b', 4096),
    ('vicuna-7b', 4096),
    ('vicuna-13b', 5120),
    ('vicuna-33b', 6656),
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
