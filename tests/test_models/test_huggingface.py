import logging
import pytest

from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.models.huggingface import HuggingfaceSubject

logging.basicConfig(level=logging.INFO)


class TestHuggingfaceSubject:

    # @pytest.mark.memory_intense
    # def test_reading_time(self):
    #     """
    #     This is a simple test that takes in text = 'the quick brown fox', and tests reading time (estimated using
    #     perplexity).
    #     This test is a stand-in prototype to check if our model definitions are correct.
    #     """
    #     model = HuggingfaceSubject(model_id='distilgpt2',
    #                                region_layer_mapping={}
    #                                )
    #     text = ['the quick brown fox', 'jumps over', 'the lazy']
    #     logging.info(f'Running {model.identifier()} with text "{text}"')
    #     model.perform_behavioral_task(task=ArtificialSubject.Task.reading_times)
    #     reading_times = model.digest_text(text)['behavior'].values
    #     print(model.identifier(), reading_times)
    #     assert reading_times == 5.7483068

    # @pytest.mark.parametrize('model_identifier, expected_next_word', [
    #     pytest.param('bert-base-uncased', '.', marks=pytest.mark.memory_intense),
    #     pytest.param('gpt2-xl', ' jumps', marks=pytest.mark.memory_intense),
    #     ('distilgpt2', 'es'),
    # ])
    # def test_next_word(self, model_identifier, expected_next_word):
    #     """
    #     This is a simple test that takes in text = 'the quick brown fox', and tests the next word.
    #     This test is a stand-in prototype to check if our model definitions are correct.
    #     """
    #
    #     model = HuggingfaceSubject(model_id=model_identifier,
    #                                region_layer_mapping={}
    #                                )
    #     text = 'the quick brown fox'
    #     logging.info(f'Running {model.identifier()} with text "{text}"')
    #     model.perform_behavioral_task(task=ArtificialSubject.Task.next_word)
    #     next_word = model.digest_text(text)['behavior'].values
    #     assert next_word == expected_next_word
    #
    # @pytest.mark.parametrize('model_identifier, expected_next_word', [
    #     pytest.param('bert-base-uncased', ['.', '.', '.'], marks=pytest.mark.memory_intense),
    #     pytest.param('gpt2-xl', [' jumps', ' the', ','], marks=pytest.mark.memory_intense),
    #     ('distilgpt2', ['es', ' the', ','] ),
    # ])
    # def test_behavior_multiple_texts(self, model_identifier, expected_next_word):
    #     """
    #     This is a simple test that takes in text = ['the quick brown fox', 'jumps over', 'the lazy'], and tests the
    #     next word for each sentence in the list.
    #     This test is a stand-in prototype to check if our model definitions are correct.
    #     """
    #     model = HuggingfaceSubject(model_id=model_identifier,
    #                                region_layer_mapping={}
    #                                )
    #     text = ['the quick brown fox', 'jumps over', 'the lazy']
    #     logging.info(f'Running {model.identifier()} with text "{text}"')
    #     model.perform_behavioral_task(task=ArtificialSubject.Task.next_word)
    #     next_words = [word['behavior'].values for word in  model.digest_text(text)]
    #     print(model.identifier(), next_words)
    #     assert next_words == expected_next_word

    def test_representation_multiple_texts(self):
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
                                       recording_type=ArtificialSubject.RecordingType.spikerate_exact)
        representations = [word['neural'] for word in  model.digest_text(text)]
        assert [len(representation['presentation']) for representation in representations] == [1,1,1]
        assert [representation['context'].squeeze() for representation in representations] == text
        assert [len(representation['neuroid']) for representation in representations] == [768, 768, 768]

    # @pytest.mark.memory_intense
    # def test_representation_one_text_single_target(self):
    #     """
    #     This is a simple test that takes in text = 'the quick brown fox', and asserts that the `distilgpt2` layer
    #     indexed by `representation_layer` has 1 text presentation and 768 neurons. This test is a stand-in prototype to
    #     check if our model definitions are correct.
    #     """
    #     model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
    #         ArtificialSubject.RecordingTarget.language_system: 'transformer.h.0.ln_1'})
    #     text = 'the quick brown fox'
    #     logging.info(f'Running {model.identifier()} with text "{text}"')
    #     model.perform_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
    #                                    recording_type=ArtificialSubject.RecordingType.spikerate_exact)
    #     representations = model.digest_text(text)['neural']
    #     assert len(representations['presentation']) == 1
    #     assert representations['context'].squeeze() == text
    #     assert len(representations['neuroid']) == 768
    #     logging.info(f'representation shape is correct: {representations.shape}')
    #
    # @pytest.mark.memory_intense
    # def test_representation_one_text_two_targets(self):
    #     model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
    #         ArtificialSubject.RecordingTarget.language_system_left_hemisphere: 'transformer.h.0.ln_1',
    #         ArtificialSubject.RecordingTarget.language_system_right_hemisphere: 'transformer.h.1.ln_1'})
    #     text = 'the quick brown fox'
    #     logging.info(f'Running {model.identifier()} with text "{text}"')
    #     model.perform_neural_recording(
    #         recording_target=ArtificialSubject.RecordingTarget.language_system_left_hemisphere,
    #         recording_type=ArtificialSubject.RecordingType.spikerate_exact)
    #     model.perform_neural_recording(
    #         recording_target=ArtificialSubject.RecordingTarget.language_system_right_hemisphere,
    #         recording_type=ArtificialSubject.RecordingType.spikerate_exact)
    #     representations = model.digest_text(text)['neural']
    #     assert len(representations['presentation']) == 1
    #     assert representations['context'].squeeze() == text
    #     assert len(representations['neuroid']) == 768 * 2
    #     assert set(representations['region'].values) == {
    #         ArtificialSubject.RecordingTarget.language_system_left_hemisphere,
    #         ArtificialSubject.RecordingTarget.language_system_right_hemisphere}
    #     logging.info(f'representation shape is correct: {representations.shape}')
    #
    # # TODO: add test with long text input, e.g. thousands of words,
    # #  to see if we need batching, and to stress-test token alignment
    #
    # # TODO: add test with multiple passage input and representation retrieval, e.g. ['the', 'quick brown', 'fox']
