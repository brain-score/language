import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from brainmodel import BrainModel
# from brainscore_language.brainmodel import BrainModel
from benchmarks.simple_next_word import predict_next_word

class HuggingfaceModel(BrainModel):

    print('start')

    def __init__(
        self,
        model_id,
    ):
        """
        Args:
            model_id (str): the model id i.e. name
        """

        # from transformers import AutoConfig, AutoModel, AutoTokenizer
        from transformers import logging as transformers_logging
        from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

        # from transformers import pipeline

        transformers_logging.set_verbosity_error()

        self.model_id = model_id
        # self.device = device or get_torch_device()
        # self.config = AutoConfig.from_pretrained(self.model_id)
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.model_id, multiprocessing=True
        # )
        # self.model = AutoModel.from_pretrained(self.model_id, config=self.config)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_id)

        return predict_next_word(input = 'what is the point',
                      tokenizer=self.tokenizer,
                      model=self.model)

        print('init complete')

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


temp = HuggingfaceModel(model_id="allenai/t5-small-next-word-generator-qoogle")
# temp = HuggingfaceModel(model_id='distilgpt2')
print(temp.identifier() )
temp.start_task(BrainModel.Task.next_word)
text = 'the quick brown fox'
# next_word = temp.digest_text(text)

# generator = pipeline('text-generation', model='distilgpt2')
# print(generator("Hello, I’m a language model", max_length=20, num_return_sequences=5))
# # Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.