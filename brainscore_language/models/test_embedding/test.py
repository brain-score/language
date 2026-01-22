"""
Tests for the test-embedding model.
"""
import numpy as np
import pytest
from brainscore_language import load_model, ArtificialSubject, score


def test_neural():
    """Test that the model can produce neural recordings."""
    model = load_model('test-embedding')
    text = ['the quick brown fox', 'jumps over', 'the lazy dog']
    model.start_neural_recording(
        recording_target=ArtificialSubject.RecordingTarget.language_system,
        recording_type=ArtificialSubject.RecordingType.fMRI
    )
    representations = model.digest_text(text)['neural']
    assert len(representations['presentation']) == 3
    np.testing.assert_array_equal(representations['stimulus'], text)
    assert len(representations['neuroid']) == 768  # distilgpt2 hidden size


def test_score():
    """Test that the model can be scored on a benchmark."""
    # Use a small, fast benchmark for testing
    result = score(
        model_identifier='test-embedding',
        benchmark_identifier='Pereira2018.243sentences-linear'
    )
    # Just check that we get a valid score (not NaN or None)
    assert result is not None
    assert not np.isnan(result)
    assert 0 <= result <= 1  # Correlation scores should be in [0, 1]
