import unittest
from brainscore_language.models.huggingface import HuggingfaceSubject
from artificial_subject import ArtificialSubject
import logging
logging.basicConfig(level=logging.INFO)

class TestHuggingfaceSubject(unittest.TestCase):
    def test_next_word(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import logging as transformers_logging

        model = HuggingfaceSubject(model_id='distilgpt2',
                                 model_class=AutoModelForCausalLM,
                                 tokenizer_class=AutoTokenizer
                                 )

        logging.info(' '.join(['Running', model.identifier(), 'for next word prediction']) )
        model.perform_task(ArtificialSubject.Task.next_word)
        text = 'the quick brown fox'
        next_word = model.digest_text(text)
        assert next_word == 'es'
        logging.info( ' '.join(['next_word:', next_word]) )
