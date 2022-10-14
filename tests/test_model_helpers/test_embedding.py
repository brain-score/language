import logging
from pathlib import Path

import numpy as np
import pytest

from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.model_helpers.embedding import GensimKeyedVectorsSubject, remove_punctuation

logging.basicConfig(level=logging.INFO)


class TestGensimNeural:
    @pytest.mark.parametrize('word, expected_vector', [
        ('the', [-0.082752, 0.67204, -0.149879]),
        ('quick', [1, 2, 3]),
        ('brown', [3, 2, 1]),
        ('fox', [8, 6, 1]),
        ('jumped', [0, 2, 3]),
        ('over', [0, 0, 0]),
    ])
    def test_by_word(self, word, expected_vector):
        model = GensimKeyedVectorsSubject(
            identifier='dummy', weights_file=Path(__file__).parent / 'mini_embeddings.word2vec', vector_size=3)
        model.perform_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                       recording_type=ArtificialSubject.RecordingType.fMRI)
        representations = model.digest_text(word)['neural']
        assert len(representations['presentation']) == 1
        assert representations['stimulus'].squeeze() == word
        assert len(representations['neuroid']) == 3

    def test_multi_word(self):
        model = GensimKeyedVectorsSubject(
            identifier='dummy', weights_file=Path(__file__).parent / 'mini_embeddings.word2vec', vector_size=3)
        text = 'the quick brown fox'
        model.perform_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                       recording_type=ArtificialSubject.RecordingType.fMRI)
        representations = model.digest_text(text)['neural']
        assert len(representations['presentation']) == 1
        assert representations['stimulus'].squeeze() == text
        assert len(representations['neuroid']) == 3

    def test_list_input(self):
        model = GensimKeyedVectorsSubject(
            identifier='dummy', weights_file=Path(__file__).parent / 'mini_embeddings.word2vec', vector_size=3)
        text = ['the quick', 'brown fox', 'jumps over', 'the lazy dog']
        model.perform_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                       recording_type=ArtificialSubject.RecordingType.fMRI)
        representations = model.digest_text(text)['neural']
        assert len(representations['presentation']) == 4
        np.testing.assert_array_equal(representations['stimulus'], text)
        assert len(representations['neuroid']) == 3

    def test_one_text_two_targets(self):
        model = GensimKeyedVectorsSubject(
            identifier='dummy', weights_file=Path(__file__).parent / 'mini_embeddings.word2vec', vector_size=3)
        text = 'the quick brown fox'
        model.perform_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system_left_hemisphere,
            recording_type=ArtificialSubject.RecordingType.fMRI)
        model.perform_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system_right_hemisphere,
            recording_type=ArtificialSubject.RecordingType.fMRI)
        representations = model.digest_text(text)['neural']
        assert len(representations['presentation']) == 1
        assert representations['stimulus'].squeeze() == text
        assert len(representations['neuroid']) == 3 * 2
        assert set(representations['recording_target'].values) == {
            ArtificialSubject.RecordingTarget.language_system_left_hemisphere,
            ArtificialSubject.RecordingTarget.language_system_right_hemisphere}


class TestPunctuation:
    def test_removes_dot_at_end(self):
        assert remove_punctuation('word.') == 'word'

    def test_removes_dot_middle(self):
        assert remove_punctuation('wor.d') == 'word'

    def test_removes_comma_at_end(self):
        assert remove_punctuation('word,') == 'word'

    def test_removes_colon_at_end(self):
        assert remove_punctuation('word:') == 'word'

    def test_removes_questionmark_at_end(self):
        assert remove_punctuation('word?') == 'word'

    def test_removes_exclamationmark_at_end(self):
        assert remove_punctuation('word!') == 'word'

    def test_not_removes_dash(self):
        assert remove_punctuation('wo-rd') == 'wo-rd'

    def test_not_removes_apostrophe(self):
        assert remove_punctuation("they're") == "they're"
