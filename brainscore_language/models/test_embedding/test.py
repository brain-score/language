"""
Tests for the test-embedding model.
"""
import numpy as np
import pytest
from brainscore_language import load_model, ArtificialSubject, score

from brainscore_language.models.test_embedding import SimpleTestEmbedding


class TestSimpleTestEmbedding:
    def test_embedding_dimension(self):
        embedding = SimpleTestEmbedding(embedding_size=50)['the']
        assert len(embedding) == 50

    def test_consistent(self):
        embedder = SimpleTestEmbedding(50)
        embedding1 = embedder['the']
        embedding2 = embedder['the']
        assert np.array_equal(embedding1, embedding2)

    def test_unique(self):
        embedder = SimpleTestEmbedding(50)
        embedding1 = embedder['the']
        embedding2 = embedder['fox']
        assert not np.array_equal(embedding1, embedding2)


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
    assert len(representations['neuroid']) == 50  # embedding_size


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
