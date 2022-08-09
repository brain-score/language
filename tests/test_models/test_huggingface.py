import logging
import unittest
import pytest


from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.models.huggingface import HuggingfaceSubject

logging.basicConfig(level=logging.INFO)


class TestHuggingfaceSubject(unittest.TestCase):


    @pytest.mark.memory_intense
    def test_next_word_mask_bert_base_uncased(self):
        """
        This is a simple test that takes in text = 'the quick brown fox', and asserts
        that the next word predicted is '.'
        This test is a stand-in prototype to check if our model definitions are correct.
        """
        model = HuggingfaceSubject(model_id='bert-base-uncased',
                                   region_layer_mapping={}
                                    )

        logging.info(' '.join(['Running', model.identifier(), 'for next word prediction test']) )
        text = 'the quick brown fox jumps over the lazy [MASK]'
        model.perform_behavioral_task(
                           task=ArtificialSubject.Task.next_word,
                           )
        next_word = model.digest_text(text)['behavior'].values
        assert next_word == '.'

    @pytest.mark.memory_intense
    def test_next_word_gpt2_xl(self):
        """
        This is a simple test that takes in text = 'the quick brown fox', and asserts
        that the next word predicted is ' jumps'.
        This test is a stand-in prototype to check if our model definitions are correct.
        """

        model = HuggingfaceSubject(model_id='gpt2-xl',
                                   region_layer_mapping={}
                                    )
        text = 'the quick brown fox'
        logging.info(f'Running {model.identifier()} with text "{text}"')
        model.perform_behavioral_task(task=ArtificialSubject.Task.next_word)
        next_word = model.digest_text(text)['behavior'].values
        assert next_word == ' jumps'


    def test_next_word_distilgpt2(self):
        """
        This is a simple test that takes in text = 'the quick brown fox',
        and asserts that the next word predicted is 'es'.
        """
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
        text = 'the quick brown fox'
        logging.info(f'Running {model.identifier()} with text "{text}"')
        model.perform_behavioral_task(task=ArtificialSubject.Task.next_word)
        next_word = model.digest_text(text)['behavior'].values
        assert next_word == 'es'

    @pytest.mark.memory_intense
    def test_representation_one_text_single_target(self):
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
                                       recording_type=ArtificialSubject.RecordingType.spikerate_exact)
        representations = model.digest_text(text)['neural']
        assert len(representations['presentation']) == 1
        assert representations['context'].squeeze() == text
        assert len(representations['neuroid']) == 768
        logging.info(f'representation shape is correct: {representations.shape}')

    @pytest.mark.memory_intense
    def test_representation_one_text_two_targets(self):
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system_left_hemisphere: 'transformer.h.0.ln_1',
            ArtificialSubject.RecordingTarget.language_system_right_hemisphere: 'transformer.h.1.ln_1'})
        text = 'the quick brown fox'
        logging.info(f'Running {model.identifier()} with text "{text}"')
        model.perform_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system_left_hemisphere,
                                       recording_type=ArtificialSubject.RecordingType.spikerate_exact)
        model.perform_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system_right_hemisphere,
                                       recording_type=ArtificialSubject.RecordingType.spikerate_exact)
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
