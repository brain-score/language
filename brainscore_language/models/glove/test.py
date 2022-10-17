import numpy as np

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject
import pytest


@pytest.mark.travis_slow
def test_neural():
    model_identifier = 'glove-840b'
    expected_feature_size = 300
    model = load_model(model_identifier)
    text = ['the quick brown fox', 'jumps over', 'the lazy', 'dog']
    model.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                 recording_type=ArtificialSubject.RecordingType.fMRI)
    representations = model.digest_text(text)['neural']
    assert len(representations['presentation']) == 4
    np.testing.assert_array_equal(representations['stimulus'], text)
    assert len(representations['neuroid']) == expected_feature_size
