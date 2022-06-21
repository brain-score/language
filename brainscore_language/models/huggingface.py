import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from brainmodel import BrainModel
from benchmarks.predict_word_bag import predict_next_word

class HuggingfaceModel(BrainModel):

    def __init__(
        self,
        model_id,
    ):
        """
        Args:
            model_id (str): the model id i.e. name
        """

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM
        from transformers import logging as transformers_logging

        transformers_logging.set_verbosity_error()

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

    def identifier(self):
        return self.model_id

    def digest_text(self, todostimuli):
        return self.inference(input = todostimuli,
                              tokenizer=self.tokenizer,
                              model=self.model)

    def start_recording(self, recording_target: BrainModel.RecordingTarget):
        raise NotImplementedError()

    def start_task(self, task: BrainModel.Task):
        task_function_mapping_dict = {
            BrainModel.Task.next_word: predict_next_word
        }
        self.inference = task_function_mapping_dict[task]
