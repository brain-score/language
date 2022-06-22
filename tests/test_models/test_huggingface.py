import unittest
from brainscore_language.models.huggingface import HuggingfaceModel
from brainmodel import InSilicoModel
import logging
logging.basicConfig(level=logging.INFO)

class TestHuggingfaceModel(unittest.TestCase):
    def test_next_word(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import logging as transformers_logging

        model = HuggingfaceModel(model_id='distilgpt2',
                                 model=AutoModelForCausalLM,
                                 tokenizer=AutoTokenizer
                                 )

        logging.info(' '.join(['Running', model.identifier(), 'for next word prediction']) )
        model.start_task(InSilicoModel.Task.next_word)
        text = 'the quick brown fox'
        next_word = model.digest_text(text)
        assert next_word == 'es'
        logging.info( ' '.join(['next_word:', next_word]) )
