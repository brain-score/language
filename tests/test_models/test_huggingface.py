import logging
import unittest
from brainscore_language.models.huggingface import HuggingfaceSubject
from artificial_subject import ArtificialSubject
import logging
logging.basicConfig(level=logging.INFO)

class TestHuggingfaceSubject(unittest.TestCase):
    def test_next_word(self):
        """
        This is a simple test that takes in text = 'the quick brown fox', and asserts that the next word predicted is 'es'. This test is a stand-in prototype to check if our model definitions are correct.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = HuggingfaceSubject(model_id='distilgpt2',
                                    model_class=AutoModelForCausalLM,
                                    tokenizer_class=AutoTokenizer,
                                    representation_layer=1
                                    )

        logging.info(' '.join(['Running', model.identifier(), 'for next word prediction test']) )
        text = 'the quick brown fox'
        model.perform_task(stimuli=text,
                           task=ArtificialSubject.Task.next_word)
        model.digest_text()
        assert model.next_word == 'es'

    # def test_representation(self):
    #     """
    #     This is a simple test that takes in text = 'the quick brown fox', and asserts that the `distilgpt2` layer
    #     indexed by `representation_layer` is of shape torch.Size([1,4,768]). This test is a stand-in prototype to check if our model definitions are correct.
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
    #     text = 'the quick brown fox'
    #     model.perform_task(stimuli=text,
    #                        task=ArtificialSubject.Task.representation)
    #     model.digest_text()
    #
    #     layer_to_compare_to_brain = model.representation[model.representation_layer]
    #     assert layer_to_compare_to_brain.shape == torch.Size([1,4,768])
    #     logging.info(' '.join(['representation shape is correct:', str(layer_to_compare_to_brain.shape) ]))
