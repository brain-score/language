import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from artificial_subject import ArtificialSubject
from benchmarks.predict_word_bag import predict_next_word

from transformers import AutoModel, AutoTokenizer

class HuggingfaceSubject(ArtificialSubject):

    def __init__(
            self,
            model_id: str,
            reprensetation_layer: int,
            model_class=AutoModelForCausalLM,
            tokenizer_class=AutoTokenizer,
    ):
        """
            :param model_id (str): the model id i.e. name
            :param model (AutoModel): the model to run inference from e.g. from transformers import AutoModelForCausalLM
            :param tokenizer (AutoTokenizer): the model's associated tokenizer
        """

        self.model_id = model_id
        self.model = model_class.from_pretrained(self.model_id)
        self.tokenizer = tokenizer_class.from_pretrained(self.model_id)
        self.representation_layer = reprensetation_layer

    def identifier(self):
        return self.model_id

    def digest_text(self, todostimuli):
        return self.inference(input=todostimuli,
                              tokenizer=self.tokenizer,
                              model=self.model)

    def get_representations(self, hidden_states):
        return hidden_states[self.representation_layer]

    def perform_task(self, task: ArtificialSubject.Task):
        task_function_mapping_dict = {
            ArtificialSubject.Task.next_word: predict_next_word
        }
        self.inference = task_function_mapping_dict[task]
