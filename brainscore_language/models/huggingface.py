import functools
from collections import OrderedDict
from typing import Union, List, Tuple, Dict
from numpy.core import defchararray
import numpy as np
import torch
from brainio.assemblies import DataAssembly, NeuroidAssembly, BehavioralAssembly
from torch.utils.hooks import RemovableHandle
from transformers import AutoModelForCausalLM, AutoTokenizer

from brainscore_language.artificial_subject import ArtificialSubject


class HuggingfaceSubject(ArtificialSubject):
    def __init__(
            self,
            model_id: str,
            region_layer_mapping: dict,
            model=None,
            tokenizer=None,
    ):
        """
            :param model_id: the model id i.e. name
            :param region_layer_mapping
            :param model: the model to run inference from. Using `AutoModelForCausalLM.from_pretrained` if `None`.
            :param tokenizer: the model's associated tokenizer. Using `AutoTokenizer.from_pretrained` if `None`.
        """

        self.model_id = model_id
        self.region_layer_mapping = region_layer_mapping
        self.basemodel = model if model is not None else AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(self.model_id)

        self.behavioral_task = None
        self.neural_recordings = []

        self.task_function_mapping_dict = {
            ArtificialSubject.Task.next_word: self.predict_next_word,
            ArtificialSubject.Task.reading_times: self.estimate_reading_times,
        }

    def identifier(self):
        return self.model_id

    def perform_behavioral_task(self, task: ArtificialSubject.Task):
        self.behavioral_task = task
        self.output_to_behavior = self.task_function_mapping_dict[task]

    def perform_neural_recording(self,
                                 recording_target: ArtificialSubject.RecordingTarget,
                                 recording_type: ArtificialSubject.RecordingType):
        self.neural_recordings.append((recording_target, recording_type))

    def digest_text(self, text: Union[str, List[str]]) -> Dict[str, DataAssembly]:

        """
        :param text: the text to be used for inference e.g. "the quick brown fox"
        :return: assembly of either behavioral output or internal neural representations
        TODO: every function call, context/state of model is reset
        """
        if type(text) == list:
            text = ' '.join(text)

        # tokenize
        self.tokenized_inputs = self.tokenizer(text, return_tensors="pt")

        # prepare recording hooks
        hooks = []
        layer_representations = OrderedDict()
        for (recording_target, recording_type) in self.neural_recordings:
            layer_name = self.region_layer_mapping[recording_target]
            layer = self._get_layer(layer_name)
            hook = self._register_hook(layer, name=(recording_target, layer_name), target_dict=layer_representations)
            hooks.append(hook)

        # run and remove hooks
        with torch.no_grad():
            base_output = self.basemodel(**self.tokenized_inputs)
        for hook in hooks:
            hook.remove()

        # format output
        output = {'behavior': None, 'neural': None}
        stimuli_coords = {
            'context': ('presentation', [text] if isinstance(text, str) else text),
            # TODO: It will be hard to build a unique stimulus id (which is used for e.g. cross-validation in metrics),
            #  especially when we don't know if other text with the same words will be presented later on.
            #  Maybe we need a StimulusSet class with metadata instead of just text strings after all?
            'stimulus_id': ('presentation', [0] if isinstance(text, str) else np.arange(len(text))),
        }

        if self.behavioral_task:
            behavioral_output = self.output_to_behavior(base_output= base_output)
            behavior = BehavioralAssembly(
                [behavioral_output],
                coords=stimuli_coords,
                dims=['presentation']
            )
            output['behavior'] = behavior
        if self.neural_recordings:
            representation_values = np.concatenate([
                # use last token (-1) of values[batch, token, unit] to represent passage.
                # TODO: this is a choice and needs to be marked as such, and maybe an option given to the user
                # TODO: likely need to be clever about this when there are multiple passages
                values[:, -1:, :].squeeze(0) for values in layer_representations.values()],
                axis=-1)  # concatenate along neuron axis
            neuroid_coords = {
                'layer': ('neuroid', np.concatenate([[layer] * values.shape[-1]
                                                     for (recording_target, layer), values in
                                                     layer_representations.items()])),
                'region': ('neuroid', np.concatenate([[recording_target] * values.shape[-1]
                                                      for (recording_target, layer), values in
                                                      layer_representations.items()])),
                'neuron_number_in_layer': ('neuroid', np.concatenate(
                    [np.arange(values.shape[-1]) for values in layer_representations.values()])),
            }
            neuroid_coords['neuroid_id'] = 'neuroid', functools.reduce(defchararray.add, [
                neuroid_coords['layer'][1], '--', neuroid_coords['neuron_number_in_layer'][1].astype(str)])
            representations = NeuroidAssembly(
                representation_values,
                coords={**stimuli_coords, **neuroid_coords},
                dims=['presentation', 'neuroid'])
            output['neural'] = representations

        return output

    def estimate_reading_times(self, base_output):
        import torch.nn.functional as F
        logits = base_output.logits
        last_token_logits = logits[-1][-1]
        last_token = self.tokenized_inputs['input_ids'][0][-1]
        perplexity = F.cross_entropy(last_token_logits, last_token) # as per https://stackoverflow.com/a/59219379/1504411
        return perplexity

    def predict_next_word(self, base_output):

        logits = base_output.logits
        pred_id = torch.argmax(logits, axis=2).squeeze()
        last_model_token_inference = pred_id[-1].tolist()
        next_word = self.tokenizer.decode(last_model_token_inference)

        return next_word

    def _get_layer(self, layer_name: str) -> torch.nn.Module:
        SUBMODULE_SEPARATOR = '.'

        module = self.basemodel
        for part in layer_name.split(SUBMODULE_SEPARATOR):
            module = module._modules.get(part)
            assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
        return module

    def _register_hook(self,
                       layer: torch.nn.Module,
                       name: Union[str, Tuple[str, str]],
                       target_dict: dict) -> RemovableHandle:

        def hook_function(_layer: torch.nn.Module, _input, output: torch.Tensor, name: str = name):
            target_dict[name] = self._tensor_to_numpy(output)

        hook = layer.register_forward_hook(hook_function)
        return hook

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().data.numpy()
