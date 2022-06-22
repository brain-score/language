import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from brainmodel import InSilicoModel
from benchmarks.predict_word_bag import predict_next_word

class HuggingfaceModel(InSilicoModel):

    def __init__(
        self,
        model_id,
        model=None,
        tokenizer=None
    ):
        """
        Args:
            model_id (str): the model id i.e. name
            model (transformers.models.auto.modeling_auto): the model to run inference from e.g. from transformers import AutoModelForCausalLM
            tokenizer (transformers.models.auto.tokenization_auto): the model's associated tokenizer
        """

        self.model_id = model_id
        if model:
            self.model = model.from_pretrained(self.model_id)
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(self.model_id)

        if tokenizer:
            self.tokenizer = tokenizer.from_pretrained(self.model_id)
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

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
