import logging
import unittest
import pytest


from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.models.huggingface import HuggingfaceSubject

logging.basicConfig(level=logging.INFO)


class TestHuggingfaceSubject(unittest.TestCase):

    # @pytest.mark.memory_intense
    # def test_fill_mask_t5(self):
    #     """
    #     Text to Text approach where sentinel tokens are dropped from the original text resulting in:
    #     - input text
    #     - label text (dropped sentinel tokens)
    #     This test is a stand-in prototype to check if our model definitions are correct.
    #     """
    #     from transformers import T5ForConditionalGeneration, T5Tokenizer
    #
    #     model = HuggingfaceSubject(model_id='t5-small',
    #                                 model_class=T5ForConditionalGeneration,
    #                                 tokenizer_class=T5Tokenizer,
    #                                 )
    #
    #     logging.info(' '.join(['Running', model.identifier(), 'for next word prediction test']) )
    #     stimuli = {
    #         'input':'The <extra_id_0> walks in <extra_id_1> park',
    #         'labels': '<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>'
    #     }
    #     model.perform_task(stimuli=stimuli,
    #                        task=ArtificialSubject.Task.input_target_sequence,
    #                        )
    #     model.digest_text()
    #     print('model.extras:', model.extras)
    #     assert model.extras == '<extra_id_0> park park<extra_id_1> the<extra_id_2> park'
    #
    # @pytest.mark.memory_intense
    # def test_fill_mask_t5_and_representation(self):
    #     """
    #     Text to Text approach where sentinel tokens are dropped from the original text resulting in:
    #     - input text
    #     - label text (dropped sentinel tokens)
    #     This test is a stand-in prototype to check if our model definitions are correct.
    #     """
    #     from transformers import T5ForConditionalGeneration, T5Tokenizer
    #     import torch
    #
    #     model = HuggingfaceSubject(model_id='t5-small',
    #                                 model_class=T5ForConditionalGeneration,
    #                                 tokenizer_class=T5Tokenizer,
    #                                 )
    #
    #     logging.info(' '.join(['Running', model.identifier(), 'for next word prediction test']) )
    #     stimuli = {
    #         'input':'The <extra_id_0> walks in <extra_id_1> park',
    #         'labels': '<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>'
    #     }
    #     model.perform_task(stimuli=stimuli,
    #                        task=ArtificialSubject.Task.input_target_sequence,
    #                        recording=True,
    #                        language_system='left_hemisphere_T5'
    #                        )
    #     model.digest_text()
    #
    #     layer_to_compare_to_brain = model.get_representations()
    #     assert layer_to_compare_to_brain.shape == torch.Size([1,7,512])
    #     logging.info(' '.join(['representation shape is correct:', str(layer_to_compare_to_brain.shape) ]))
    #
    # @pytest.mark.memory_intense
    # def test_fill_mask_bert_base_uncased(self):
    #     """
    #     This is a simple test that takes in text = 'the quick brown fox', and asserts
    #     that the next word predicted is 'es'.
    #     This test is a stand-in prototype to check if our model definitions are correct.
    #     """
    #     # from transformers import AutoModelForCausalLM, AutoTokenizer
    #
    #     model = HuggingfaceSubject(model_id='bert-base-uncased',
    #                                 # model_class=AutoModelForCausalLM,
    #                                 # tokenizer_class=AutoTokenizer,
    #                                region_layer_mapping={}
    #                                 )
    #
    #     logging.info(' '.join(['Running', model.identifier(), 'for next word prediction test']) )
    #     text = 'the quick brown fox [MASK] over the lazy dog'
    #     # self.tokenizer.decode(tokenized_inputs['input_ids']): '[CLS] the quick brown fox [MASK] over the lazy dog [SEP]'"
    #     model.perform_behavioral_task(
    #                        # stimuli=text,
    #                        task=ArtificialSubject.Task.fill_mask,
    #                        )
    #     # model.digest_text()
    #     fill_mask_word = model.digest_text(text)['behavior'].values
    #     # print(model.fill_mask_word)
    #     assert model.fill_mask_word.strip() == 'took'


    def test_next_word_gpt2_xl(self):
        """
        This is a simple test that takes in text = 'the quick brown fox', and asserts
        that the next word predicted is 'es'.
        This test is a stand-in prototype to check if our model definitions are correct.
        """

        model = HuggingfaceSubject(model_id='gpt2-xl',
                                   region_layer_mapping={}
                                    )

        logging.info(' '.join(['Running', model.identifier(), 'for next word prediction test']) )
        text = 'the quick brown fox'
        model.perform_behavioral_task(
                           task=ArtificialSubject.Task.next_word,
                           )
        next_word = model.digest_text(text)['behavior'].values
        assert next_word[0].strip() == 'jumps'


    def test_next_word(self):
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
