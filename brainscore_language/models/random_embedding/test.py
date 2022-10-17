import numpy as np
import pytest
from brainscore_language import load_model, ArtificialSubject, score

from brainscore_language.models.random_embedding import WordToEmbedding


class TestWordToEmbedding:
    @pytest.mark.parametrize('embedding_size', [10, 300, 600])
    def test_embedding_dimension(self, embedding_size):
        embedding = WordToEmbedding(embedding_size)['the']
        assert len(embedding) == embedding_size

    def test_consistent(self):
        embedder = WordToEmbedding(300)
        embedding1 = embedder['the']
        embedding2 = embedder['the']
        assert np.array_equal(embedding1, embedding2)

    def test_unique(self):
        embedder = WordToEmbedding(300)
        embedding1 = embedder['the']
        embedding2 = embedder['fox']
        assert not np.array_equal(embedding1, embedding2)

    def test_ordering_consistent(self):
        embedder = WordToEmbedding(300)
        embedding1a = embedder['the']
        embedding2a = embedder['fox']
        embedding2b = embedder['fox']
        embedding1b = embedder['the']
        assert np.array_equal(embedding1a, embedding1b)
        assert np.array_equal(embedding2a, embedding2b)
        assert not np.array_equal(embedding1a, embedding2a)


def test_neural():
    model = load_model('randomembedding-1600')
    expected_feature_size = 1600
    text = ['the quick brown fox', 'jumps over', 'the lazy', 'dog', 'the lazy']
    model.perform_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                   recording_type=ArtificialSubject.RecordingType.fMRI)
    representations = model.digest_text(text)['neural']
    assert len(representations['presentation']) == 5
    np.testing.assert_array_equal(representations['stimulus'], text)
    assert len(representations['neuroid']) == expected_feature_size
    assert np.array_equal(representations.sel(part_number=2), representations.sel(part_number=4))


def test_score():
    result = score(model_identifier='randomembedding-100', benchmark_identifier='Pereira2018.243sentences-linear')
    assert result == pytest.approx(.0285022, abs=.005)
