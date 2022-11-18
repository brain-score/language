import numpy as np
import pytest

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject


@pytest.mark.memory_intense
@pytest.mark.parametrize(
    "model_identifier, expected_reading_times",
    [
        (
            "rnn-lm-ptb",
            [10.370, 15.490, 18.165, 2.902, 15.848, 6.329, 1.238, 15.902, 14.127],
        )
    ],
)
def test_reading_times(model_identifier, expected_reading_times):
    model = load_model(model_identifier)
    text = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
    reading_times = model.digest_text(text)["behavior"]
    np.testing.assert_allclose(reading_times, expected_reading_times, atol=0.01)


@pytest.mark.memory_intense
@pytest.mark.parametrize(
    "model_identifier, expected_next_words",
    [
        ("rnn-lm-ptb", [",", "the", "of"]),
    ],
)
def test_next_word(model_identifier, expected_next_words):
    model = load_model(model_identifier)
    text = ["the quick", "brown fox jumps over", "the lazy dog"]
    model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
    next_word_predictions = model.digest_text(text)["behavior"]
    np.testing.assert_array_equal(next_word_predictions, expected_next_words)


@pytest.mark.memory_intense
@pytest.mark.parametrize(
    "model_identifier, feature_size",
    [
        ("rnn-lm-ptb", 650),
    ],
)
def test_neural(model_identifier, feature_size):
    model = load_model(model_identifier)
    text = ["the quick brown fox", "jumps over", "the lazy dog"]
    model.start_neural_recording(
        recording_target=ArtificialSubject.RecordingTarget.language_system,
        recording_type=ArtificialSubject.RecordingType.fMRI,
    )
    representations = model.digest_text(text)["neural"]
    assert len(representations["presentation"]) == 3
    np.testing.assert_array_equal(representations["stimulus"], text)
    assert len(representations["neuroid"]) == feature_size
