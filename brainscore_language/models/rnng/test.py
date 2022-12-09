import numpy as np
import pytest

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject


@pytest.mark.memory_intense
@pytest.mark.parametrize(
    "model_identifier, feature_size",
    [
        ("rnn-tdg-ptb", 512),
        ("rnn-lcg-ptb", 512),
        ("rnn-tdg-ptboanc", 512),
        ("rnn-lcg-ptboanc", 512),
        ("rnn-tdg-ptboanc-1024", 1024),
        ("rnn-lcg-ptboanc-1024", 1024),
    ],
)
def test_neural(model_identifier, feature_size):
    model = load_model(model_identifier)
    text = ["the quick brown fox jumps over the lazy dog."]
    model.start_neural_recording(
        recording_target=ArtificialSubject.RecordingTarget.language_system,
        recording_type=ArtificialSubject.RecordingType.fMRI,
    )
    representations = model.digest_text(text)["neural"]
    assert len(representations["presentation"]) == 1
    np.testing.assert_array_equal(representations["stimulus"], text)
    assert len(representations["neuroid"]) == feature_size
