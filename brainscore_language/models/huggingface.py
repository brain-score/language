import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from artificial_subject import ArtificialSubject

from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingfaceSubject(ArtificialSubject):

    def __init__(
            self,
            model_id: str,
            model_class=AutoModelForCausalLM,
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
        # self.representation_layer = representation_layer

        self.next_word = None
        self.representation = None
        # self.recording = recording


    def identifier(self):
        return self.model_id

    def digest_text(self):
        return self.run_experiment()
        # for func in self.experiments_to_run:
        #     func()

    def get_representations(self):
        if not self.recording:
            import sys
            raise Exception("You cannot call `get_representations` when `recording` is False")
        return self.representation[self.layer_name]

    def perform_task(self, stimuli: str,
                     task: ArtificialSubject.Task,
                     recording=False,
                     language_system = None
                     ):
        task_function_mapping_dict = {
            ArtificialSubject.Task.next_word: self.predict_next_word,
        }

        region_layer_mapping = {'Broca': 'transformer.h.0.ln_1' #stand-in example
                                }

        self.stimuli = stimuli
        self.recording = recording
        self.run_experiment = task_function_mapping_dict[task]
        if recording:
            self.layer_name =  region_layer_mapping[language_system]

    def predict_next_word(self):
        """
        :param seq: the text to be used for inference e.g. "the quick brown fox"
        :param tokenizer: huggingface tokenizer, defined in the HuggingfaceModel class via: self.tokenizer =
        AutoTokenizer.from_pretrained(self.model_id)
        :param model: huggingface model, defined in the HuggingfaceModel class via: self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        :return: single string which reprensets the model's prediction of the next word
        """
        import torch
        from collections import OrderedDict

        tokenized_inputs = self.tokenizer(self.stimuli, return_tensors="pt")

        if self.recording:
            self.representation = OrderedDict()
            hooks = []
            layer = self.get_layer(self.layer_name)
            hook = self.register_hook(layer, self.layer_name, target_dict=self.representation)
            hooks.append(hook)

        with torch.no_grad():
            output = self.model(**tokenized_inputs, output_hidden_states=True, return_dict=True)

        if self.recording:
            for hook in hooks:
                hook.remove()

        logits = output['logits']
        pred_id = torch.argmax(logits, axis=2).squeeze()
        last_model_token_inference = pred_id[-1].tolist()
        self.next_word = self.tokenizer.decode(last_model_token_inference)

    def get_layer(self, layer_name):
        SUBMODULE_SEPARATOR = '.'

        module = self.model
        for part in layer_name.split(SUBMODULE_SEPARATOR):
            print(part)
            module = module._modules.get(part)
            assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
        return module

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            target_dict[name] = HuggingfaceSubject._tensor_to_numpy(output)

        hook = layer.register_forward_hook(hook_function)
        return hook

    @classmethod
    def _tensor_to_numpy(cls, output):
        return output.cpu().data.numpy()

"""
Done:
- move predict_next_word function to hugginface.py
- do we get a hidden state per token (words can sometimes be broken into tokens), or one for the whole sentence
- stateful output i.e. if this is a reprensentation task, return the hidden state, if this is next word, output next word, etc.

Not done:
- add a region_layer_mapping that maps from model layer names to hidden state tensor indices. 
"""

