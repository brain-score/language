import logging
import unittest
from brainscore_language.models.huggingface import HuggingfaceSubject
from artificial_subject import ArtificialSubject
import logging
logging.basicConfig(level=logging.INFO)

class TestHuggingfaceSubject(unittest.TestCase):
    def test_next_word_gpt2_xl(self):
        """
        This is a simple test that takes in text = 'the quick brown fox', and asserts
        that the next word predicted is 'es'.
        This test is a stand-in prototype to check if our model definitions are correct.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = HuggingfaceSubject(model_id='gpt2-xl',
                                    model_class=AutoModelForCausalLM,
                                    tokenizer_class=AutoTokenizer,
                                    representation_layer=1
                                    )

        logging.info(' '.join(['Running', model.identifier(), 'for next word prediction test']) )
        model.perform_task(ArtificialSubject.Task.next_word)
        text = 'the quick brown fox'
        next_word, _ = model.digest_text(text)
        print(next_word)
        assert next_word == 'jumps'
        logging.info(' '.join(['next_word:', next_word]))

    # def test_next_word_distilgpt2(self):
    #     """
    #     This is a simple test that takes in text = 'the quick brown fox', and asserts
    #     that the next word predicted is 'es'.
    #     This test is a stand-in prototype to check if our model definitions are correct.
    #     """
    #     from transformers import AutoModelForCausalLM, AutoTokenizer
    #
    #     model = HuggingfaceSubject(model_id='distilgpt2',
    #                                 model_class=AutoModelForCausalLM,
    #                                 tokenizer_class=AutoTokenizer,
    #                                 representation_layer=1
    #                                 )
    #
    #     logging.info(' '.join(['Running', model.identifier(), 'for next word prediction test']) )
    #     model.perform_task(ArtificialSubject.Task.next_word)
    #     text = 'the quick brown fox'
    #     next_word, _ = model.digest_text(text)
    #     assert next_word == 'es'
    #     logging.info(' '.join(['next_word:', next_word]))
    #
    # def test_representation(self):
    #     """
    #     This is a simple test that takes in text = 'the quick brown fox', and asserts
    #     that the `distilgpt2` layer indexed by `representation_layer` is of shape torch.Size([1,4,768]).
    #     This test is a stand-in prototype to check if our model definitions are correct.
    #     """
    #
    #     from transformers import AutoModelForCausalLM, AutoTokenizer
    #     import torch
    #
    #     model = HuggingfaceSubject(model_id='distilgpt2',
    #                                 model_class=AutoModelForCausalLM,
    #                                 tokenizer_class=AutoTokenizer,
    #                                 representation_layer=1
    #                                 )
    #
    #     logging.info(' '.join(['Running', model.identifier(), 'for representation layer shape test']) )
    #     model.perform_task(ArtificialSubject.Task.next_word)
    #     text = 'the quick brown fox'
    #     _, representations = model.digest_text(text)
    #     layer_to_compare_to_brain = representations[model.representation_layer]
    #     assert representations[model.representation_layer].shape == torch.Size([1,4,768])
    #     logging.info(' '.join(['representation shape is correct:', str(representations[model.representation_layer].shape) ]))
