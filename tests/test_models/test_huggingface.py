import logging

import numpy as np
import pytest

from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.models.huggingface import HuggingfaceSubject

logging.basicConfig(level=logging.INFO)


class TestReadingTimes:
    @pytest.mark.memory_intense
    def test_single_word(self):
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
        model.perform_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_time = model.digest_text('the')['behavior']
        assert reading_time == 0

    @pytest.mark.memory_intense
    def test_multiple_words(self):
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
        model.perform_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_time = model.digest_text('the quick brown fox')['behavior']
        assert reading_time == -44.06524

    @pytest.mark.memory_intense
    def test_list_input(self):
        model = HuggingfaceSubject(model_id='distilgpt2',
                                   region_layer_mapping={}
                                   )
        text = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy']
        logging.info(f'Running {model.identifier()} with text "{text}"')
        model.perform_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_times = model.digest_text(text)['behavior']
        np.testing.assert_allclose(
            reading_times, [0, -19.260605, -12.721411, -12.083241, -10.876629, -3.678278, -2.102749, -11.961533],
            atol=0.0001)

    @pytest.mark.memory_intense
    def test_multitoken_words(self):
        model = HuggingfaceSubject(model_id='distilgpt2',
                                   region_layer_mapping={}
                                   )
        text = ['beekepers', 'often', 'go', 'beekeeping']
        logging.info(f'Running {model.identifier()} with text "{text}"')
        model.perform_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_times = model.digest_text(text)['behavior']
        np.testing.assert_allclose(
            reading_times, [-26.090048, -16.81876, -6.711773, -19.165783], atol=0.0001)

    @pytest.mark.memory_intense
    def test_multiword_list_input(self):
        model = HuggingfaceSubject(model_id='distilgpt2',
                                   region_layer_mapping={}
                                   )
        text = ['the quick brown fox', 'jumps over', 'the lazy']
        logging.info(f'Running {model.identifier()} with text "{text}"')
        model.perform_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_times = model.digest_text(text)['behavior']
        np.testing.assert_allclose(reading_times, [-44.06524, -14.554907, -14.064276], atol=0.0001)


class TestNextWord:
    @pytest.mark.parametrize('model_identifier, expected_next_word', [
        pytest.param('bert-base-uncased', '.', marks=pytest.mark.memory_intense),
        pytest.param('gpt2-xl', ' jumps', marks=pytest.mark.memory_intense),
        ('distilgpt2', 'es'),
    ])
    def test_single_string(self, model_identifier, expected_next_word):
        """
        This is a simple test that takes in text = 'the quick brown fox', and tests the next word.
        This test is a stand-in prototype to check if our model definitions are correct.
        """

        model = HuggingfaceSubject(model_id=model_identifier,
                                   region_layer_mapping={}
                                   )
        text = 'the quick brown fox'
        logging.info(f'Running {model.identifier()} with text "{text}"')
        model.perform_behavioral_task(task=ArtificialSubject.Task.next_word)
        next_word = model.digest_text(text)['behavior'].values
        assert next_word == expected_next_word

    @pytest.mark.parametrize('model_identifier, expected_next_words', [
        pytest.param('bert-base-uncased', ['.', '.', '.'], marks=pytest.mark.memory_intense),
        pytest.param('gpt2-xl', [' jumps', ' the', ','], marks=pytest.mark.memory_intense),
        ('distilgpt2', ['es', ' the', ',']),
    ])
    def test_list_input(self, model_identifier, expected_next_words):
        """
        This is a simple test that takes in text = ['the quick brown fox', 'jumps over', 'the lazy'], and tests the
        next word for each text part in the list.
        This test is a stand-in prototype to check if our model definitions are correct.
        """
        model = HuggingfaceSubject(model_id=model_identifier,
                                   region_layer_mapping={}
                                   )
        text = ['the quick brown fox', 'jumps over', 'the lazy']
        logging.info(f'Running {model.identifier()} with text "{text}"')
        model.perform_behavioral_task(task=ArtificialSubject.Task.next_word)
        next_words = model.digest_text(text)['behavior']
        np.testing.assert_array_equal(next_words, expected_next_words)


class TestNeural:
    def test_list_input(self):
        """
        This is a simple test that takes in text = ['the quick brown fox', 'jumps over', 'the lazy'], and tests the
        representation for next word prediction for each sentence in the list.
        This test is a stand-in prototype to check if our model definitions are correct.
        """

        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system: 'transformer.h.0.ln_1'})
        text = ['the quick brown fox', 'jumps over', 'the lazy dog']
        logging.info(f'Running {model.identifier()} with text "{text}"')
        model.perform_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                       recording_type=ArtificialSubject.RecordingType.fMRI)
        representations = model.digest_text(text)['neural']
        assert len(representations['presentation']) == 3
        np.testing.assert_array_equal(representations['context'], text)
        assert len(representations['neuroid']) == 768

    @pytest.mark.memory_intense
    def test_one_text_single_target(self):
        """
        This is a simple test that takes in text = 'the quick brown fox', and asserts that the `distilgpt2` layer
        indexed by `representation_layer` has 1 text presentation and 768 neurons. This test is a stand-in prototype to
        check if our model definitions are correct.
        """
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system: 'transformer.h.0.ln_1'})
        text = 'the quick brown fox'
        logging.info(f'Running {model.identifier()} with text "{text}"')
        model.perform_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                       recording_type=ArtificialSubject.RecordingType.fMRI)
        representations = model.digest_text(text)['neural']
        assert len(representations['presentation']) == 1
        assert representations['context'].squeeze() == text
        assert len(representations['neuroid']) == 768
        logging.info(f'representation shape is correct: {representations.shape}')

    @pytest.mark.memory_intense
    def test_one_text_two_targets(self):
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system_left_hemisphere: 'transformer.h.0.ln_1',
            ArtificialSubject.RecordingTarget.language_system_right_hemisphere: 'transformer.h.1.ln_1'})
        text = 'the quick brown fox'
        logging.info(f'Running {model.identifier()} with text "{text}"')
        model.perform_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system_left_hemisphere,
            recording_type=ArtificialSubject.RecordingType.fMRI)
        model.perform_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system_right_hemisphere,
            recording_type=ArtificialSubject.RecordingType.fMRI)
        representations = model.digest_text(text)['neural']
        assert len(representations['presentation']) == 1
        assert representations['context'].squeeze() == text
        assert len(representations['neuroid']) == 768 * 2
        assert set(representations['region'].values) == {
            ArtificialSubject.RecordingTarget.language_system_left_hemisphere,
            ArtificialSubject.RecordingTarget.language_system_right_hemisphere}
        logging.info(f'representation shape is correct: {representations.shape}')

    # TODO: add test with long text input, e.g. thousands of words,
    #  to see if we need batching, and to stress-test token alignment

    # TODO: add test with multiple passage input and representation retrieval, e.g. ['the', 'quick brown', 'fox']
