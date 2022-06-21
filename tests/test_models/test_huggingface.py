import unittest
from brainscore_language.models.huggingface import HuggingfaceModel
from brainmodel import BrainModel
import logging
logging.basicConfig(level=logging.INFO)

class TestHuggingfaceModel(unittest.TestCase):
    def test_next_word(self):
        model = HuggingfaceModel(model_id='distilgpt2')
        # print('Running', model.identifier(), 'for next word prediction' )
        # logging.info('This is an info message')
        logging.info(' '.join(['Running', model.identifier(), 'for next word prediction']) )
        model.start_task(BrainModel.Task.next_word)
        text = 'the quick brown fox'
        next_word = model.digest_text(text)
        assert next_word == 'es'
        logging.info(' '.join(['next_word:', next_word]) )
