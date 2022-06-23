import logging

from brainscore_language.brainmodel import InSilicoModel
from brainscore_language.models.huggingface import HuggingfaceModel

logging.basicConfig(level=logging.INFO)


class TestHuggingfaceModel:
    def test_next_word(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = HuggingfaceModel(model_id='distilgpt2',
                                 model=AutoModelForCausalLM,
                                 tokenizer=AutoTokenizer)
        logging.info(' '.join(['Running', model.identifier(), 'for next word prediction']))
        model.start_task(InSilicoModel.Task.next_word)
        text = 'the quick brown fox'
        next_word = model.digest_text(text)
        assert next_word == 'es'
        logging.info(' '.join(['next_word:', next_word]))

# from brainscore_language.models.huggingface import distilgpt2
#
# from brainscore_language.brainmodel import BrainModel
#
#
# def test_next_word():
#     model = distilgpt2()
#     model.start_task(task=BrainModel.Task.next_word)
#     text = 'the quick brown fox'
#     next_word = model.digest_text(text)
#     assert isinstance(next_word, str)
#
#
# def test_has_interface_functions():
#     model = distilgpt2()
#     for expected_method in dir(BrainModel):
#         assert hasattr(model, expected_method)
#
#
# def test_fmri_output_format():
#     model = distilgpt2()
#     model.start_recording(recording_target=BrainModel.RecordingTarget.language_system,
#                           recording_type=BrainModel.RecordingTarget.fMRI)
#     text = 'the quick brown fox'
#     recordings = model.digest_text(text)
#     assert set(recordings.dims) == {'presentation', 'neuroid'}
#     assert recordings['text'].item() == text
#     assert len(recordings['neuroid'] > 1)
