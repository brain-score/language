import numpy as np
import pytest

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, expected_reading_times', [
    ('t5-small', [25.646585, 23.780153, 23.018826, 22.344381, 11.96658, 27.054287, 10.594951, 13.187043]),
    ('t5-base', [7.7039944e-03, 6.8635613e-02, 3.1093130e+01, 1.2913298e+02, 8.5430244e+01, 1.6261120e+01, 8.2980719e+00, 2.9535002e+01]),
    ('t5-large', [31.604916, 18.852331, 30.816673, 48.99762, 49.006733, 36.088543, 14.189968, 37.781395]),
    ('t5-xl', [ 5.2831264, 18.823713, 19.249414, 35.212494, 24.10475, 19.929758, 11.064505, 16.397375 ]),
    ('t5-xxl', [26.934216, 30.064108, 18.61358, 71.8481, 20.456089, 18.108957, 25.52297, 20.845043]),
    ('flan-t5-small', [4.626572, 5.4074254, 2.9690156, 5.98445, 12.027061, 11.096782, 16.912296, 14.794151]),
    ('flan-t5-base', [1.8610231, 1.5091983, 2.3265584, 2.5798035, 0.9352376, 2.594869, 3.4819074, 2.7790558]),
    ('flan-t5-large', [2.2994747, 4.1134634, 1.6111257, 10.103671, 11.365605, 3.37785, 1.4599704, 2.9243639]),
    ('flan-t5-xl', [2.5323708, 2.9281907, 3.2239344, 10.614168, 7.162341, 3.0385818, 2.9526176, 2.7103176]),
    ('flan-t5-xxl', [2.3222983, 2.3133714, 2.8529167, 11.162584, 6.798625, 4.742971, 2.9756427, 2.9877827]),
    ('flan-alpaca-base', [0.5997408, 1.1441187, 1.3299922, 2.1235154, 1.5477583, 0.27742645, 0.3976275, 0.21495701]),
    ('flan-alpaca-large', [0.03638878, 0.07655565, 0.02087213, 11.400998, 9.982766, 0.82122284, 0.42820516, 0.39627305]),
    ('flan-alpaca-xl', [3.2593443, 3.6223898, 3.3259575, 12.523176, 6.452489, 5.2135086, 3.7474098, 3.6356025]),
    ('flan-alpaca-xxl', [2.916435, 5.631528, 3.178902, 11.2796755, 5.902015, 2.294983, 2.8577528, 2.9340065]),
    ('flan-gpt4all-xl', [6.95467, 8.141007, 6.8901677, 7.149359, 7.247072, 7.390025, 5.7526765, 4.9763246]),
    ('flan-sharegpt-xl', [3.0441425, 2.9028635, 3.034965, 5.7231064, 2.282282, 2.5237873, 1.0039636, 1.014216]),
    ('flan-alpaca-gpt4-xl', [5.705884, 6.2532945, 5.6363673, 12.22221, 6.067267, 4.2973313, 4.1460104, 5.088393]),
])
def test_reading_times(model_identifier, expected_reading_times):
    model = load_model(model_identifier)
    text = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy']
    model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
    reading_times = model.digest_text(text)['behavior']
    np.testing.assert_allclose(reading_times, expected_reading_times, atol=0.01)


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, expected_next_words', [
    ('t5-small', ['in', 'in', 'in']),
    ('t5-base', ['<extra_id_27>', '</s>', '<extra_id_27>']),
    ('t5-large', ['<extra_id_11>', '<extra_id_11>', '<extra_id_11>']),
    ('t5-xl', ['', '', '']),
    ('t5-xxl', ['', 'ES', ',']),
    ('flan-t5-small', ['...', '...', '...']),
    ('flan-t5-base', ['</s>', '...', '</s>']),
    ('flan-t5-large', ['', '', '']),
    ('flan-t5-xl', ['', '...', '</s>']),
    ('flan-t5-xxl', ['</s>', '.', '.']),
    ('flan-alpaca-base', ['</s>', '</s>', '</s>']),
    ('flan-alpaca-large', ['', '</s>', '</s>']),
    ('flan-alpaca-xl', ['', '.', '.']),
    ('flan-alpaca-xxl', ['.', '.', '.']),
    ('flan-gpt4all-xl', ['', '', '']),
    ('flan-sharegpt-xl', ['the', '</s>', '</s>']),
    ('flan-alpaca-gpt4-xl', ['', '</s>', '</s>']),
])
def test_next_word(model_identifier, expected_next_words):
    model = load_model(model_identifier)
    text = ['the quick brown fox', 'jumps over', 'the lazy']
    model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
    next_word_predictions = model.digest_text(text)['behavior']
    np.testing.assert_array_equal(next_word_predictions, expected_next_words)


@pytest.mark.memory_intense
@pytest.mark.parametrize('model_identifier, feature_size', [
    ('t5-small', 512),
    ('t5-base', 768),
    ('t5-large', 1024),
    ('t5-xl', 2048),
    ('t5-xxl', 4096),
    ('flan-t5-small', 512),
    ('flan-t5-base', 768),
    ('flan-t5-large', 1024),
    ('flan-t5-xl', 2048),
    ('flan-t5-xxl', 4096),
    ('flan-alpaca-base', 768),
    ('flan-alpaca-large', 1024),
    ('flan-alpaca-xl', 2048),
    ('flan-alpaca-xxl', 4096),
    ('flan-gpt4all-xl', 2048),
    ('flan-sharegpt-xl', 2048),
    ('flan-alpaca-gpt4-xl', 2048),
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
