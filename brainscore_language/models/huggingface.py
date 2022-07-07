import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from artificial_subject import ArtificialSubject

from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingfaceSubject(ArtificialSubject):

    def __init__(
            self,
            model_id: str,
            representation_layer: int,
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
        self.representation_layer = representation_layer

        self.next_word = None
        self.representation = None


    def identifier(self):
        return self.model_id

    def digest_text(self):
        # return self.run_experiment()
        for func in self.experiments_to_run:
            func()

    def get_representations(self):
        # return hidden_states[self.representation_layer]
        return self.representation

    def perform_task(self, stimuli: str, task: ArtificialSubject.Task):
        task_function_mapping_dict = {
            ArtificialSubject.Task.next_word: [self.predict_next_word],
            ArtificialSubject.Task.representation: [self.predict_next_word, self.get_representations]
        }

        self.stimuli = stimuli
        self.experiments_to_run = task_function_mapping_dict[task]

    def predict_next_word(self):
        """
        :param seq: the text to be used for inference e.g. "the quick brown fox"
        :param tokenizer: huggingface tokenizer, defined in the HuggingfaceModel class via: self.tokenizer =
        AutoTokenizer.from_pretrained(self.model_id)
        :param model: huggingface model, defined in the HuggingfaceModel class via: self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        :return: single string which reprensets the model's prediction of the next word
        """
        import torch

        tokenized_inputs = self.tokenizer(self.stimuli, return_tensors="pt")
        output = self.model(**tokenized_inputs, output_hidden_states=True, return_dict=True)
        self.representation = output["hidden_states"]

        logits = output['logits']
        pred_id = torch.argmax(logits, axis=2).squeeze()
        last_model_token_inference = pred_id[-1].tolist()
        self.next_word = self.tokenizer.decode(last_model_token_inference)


"""
Done:
- move predict_next_word function to hugginface.py
- do we get a hidden state per token (words can sometimes be broken into tokens), or one for the whole sentence
- stateful output i.e. if this is a reprensentation task, return the hidden state, if this is next word, output next word, etc.

Not done:
- add a region_layer_mapping that maps from model layer names to hidden state tensor indices. 
"""