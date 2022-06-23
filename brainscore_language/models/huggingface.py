from brainscore_language.brainmodel import InSilicoModel

from .predict_word_bag import predict_next_word

from transformers import AutoModel, AutoTokenizer

class HuggingfaceModel(InSilicoModel):
    def __init__(
            self,
            model_id: str,
            model_class=AutoModel,
            tokenizer_class=AutoTokenizer
    ):
        """
            :param model_id (str): the model id i.e. name
            :param model (AutoModel): the model to run inference from e.g. from transformers import AutoModelForCausalLM
            :param tokenizer (AutoTokenizer): the model's associated tokenizer
        """

        self.model_id = model_id
        self.model = model_class.from_pretrained(self.model_id)
        self.tokenizer = tokenizer_class.from_pretrained(self.model_id)

    def identifier(self):
        return self.model_id

    def digest_text(self, todostimuli):
        return self.inference(input=todostimuli,
                              tokenizer=self.tokenizer,
                              model=self.model)

    def start_recording(self, recording_target: InSilicoModel.RecordingTarget):
        raise NotImplementedError()

    def start_task(self, task: InSilicoModel.Task):
        task_function_mapping_dict = {
            InSilicoModel.Task.next_word: predict_next_word
        }
        self.inference = task_function_mapping_dict[task]
