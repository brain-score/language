from brainmodel import InSilicoModel

from .predict_word_bag import predict_next_word


class HuggingfaceModel(InSilicoModel):
    def __init__(
            self,
            model_id: str,
            model,
            tokenizer
    ):
        """
        :param model_id: the model identifier / name
        """

        self.model_id = model_id
        self.model = model.from_pretrained(self.model_id)
        self.tokenizer = tokenizer.from_pretrained(self.model_id)
        self.inference = None

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
