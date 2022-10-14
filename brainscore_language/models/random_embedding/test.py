import numpy as np
import pytest

from brainscore_language.models.random_embedding import word_to_embedding


class TestWordToEmbedding:
    @pytest.mark.parametrize('embedding_size', [10, 300, 600])
    def test_embedding_dimension(self, embedding_size):
        embedding = word_to_embedding('the', embedding_size)
        assert len(embedding) == embedding_size

    def test_consistent(self):
        embedding1 = word_to_embedding('the', 300)
        embedding2 = word_to_embedding('the', 300)
        assert np.array_equal(embedding1, embedding2)

    def test_unique(self):
        embedding1 = word_to_embedding('the', 300)
        embedding2 = word_to_embedding('fox', 300)
        assert not np.array_equal(embedding1, embedding2)

    def test_ordering_consistent(self):
        embedding1a = word_to_embedding('the', 300)
        embedding2a = word_to_embedding('fox', 300)
        embedding2b = word_to_embedding('fox', 300)
        embedding1b = word_to_embedding('the', 300)
        assert np.array_equal(embedding1a, embedding1b)
        assert np.array_equal(embedding2a, embedding2b)
        assert not np.array_equal(embedding1a, embedding2a)
