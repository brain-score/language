import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from brainmodel import InSilicoModel
from benchmarks.predict_word_bag import predict_next_word

from transformers import AutoModel, AutoTokenizer

class HuggingfaceModel(InSilicoModel):

    def __init__(
        self,
        model_id,
        model_class=AutoModel,
        tokenizer_class=AutoTokenizer
    ):
        """
        Args:
            model_id (str): the model id i.e. name
            model (AutoModel): the model to run inference from e.g. from transformers import AutoModelForCausalLM
            tokenizer (AutoTokenizer): the model's associated tokenizer
        """

        self.model_id = model_id
        self.model = model_class.from_pretrained(self.model_id)
        self.tokenizer = tokenizer_class.from_pretrained(self.model_id)

    def identifier(self):
        return self.model_id

    def digest_text(self, todostimuli):
        return self.inference(input = todostimuli,
                              tokenizer=self.tokenizer,
                              model=self.model)

    def start_recording(self, recording_target: InSilicoModel.RecordingTarget):
        raise NotImplementedError()

    def start_task(self, task: InSilicoModel.Task):
        task_function_mapping_dict = {
            InSilicoModel.Task.next_word: predict_next_word
        }
        self.inference = task_function_mapping_dict[task]
