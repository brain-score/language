from collections import OrderedDict
from typing import Union, List, Tuple, Dict

import numpy as np
import torch
from brainio.assemblies import DataAssembly, NeuroidAssembly
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

    def identifier(self):
        return self.model_id

    def perform_behavioral_task(self, task: ArtificialSubject.Task):
        self.behavioral_task = task

    def perform_neural_recording(self,
                                 recording_target: ArtificialSubject.RecordingTarget,
                                 recording_type: ArtificialSubject.RecordingType):
        self.neural_recordings.append((recording_target, recording_type))

    def digest_text(self, text: Union[str, List[str]]) -> Dict[str, DataAssembly]:
        """
        :param text: the text to be used for inference e.g. "the quick brown fox"
        :return: assembly of either behavioral output or internal neural representations
        """
        # tokenize
        tokenized_inputs = self.tokenizer(text, return_tensors="pt")

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
            base_output = self.basemodel(**tokenized_inputs)
        for hook in hooks:
            hook.remove()

        # format output
        output = {'behavior': None, 'neural': None}
        if self.behavioral_task:
            logits = base_output.logits
            pred_id = torch.argmax(logits, axis=2).squeeze()
            last_model_token_inference = pred_id[-1].tolist()
            next_word = self.tokenizer.decode(last_model_token_inference)
            output['behavior'] = next_word
        if self.neural_recordings:
            representation_values = np.concatenate([values.squeeze(0) for values in layer_representations.values()],
                                                   axis=-1)  # concatenate along neuron axis
            representations = NeuroidAssembly(
                representation_values,
                coords={
                    # TODO: I don't think splitting on space is reliable for all models, it depends on tokenization
                    'stimuli': ('presentation', text.split(' ')),
                    'stimulus_number': ('presentation', np.arange(len(text.split(' ')))),
                    'layer': ('neuroid', np.concatenate([[layer] * values.shape[-1]
                                                         for (recording_target, layer), values in
                                                         layer_representations.items()])),
                    'region': ('neuroid', np.concatenate([[recording_target] * values.shape[-1]
                                                          for (recording_target, layer), values in
                                                          layer_representations.items()])),
                    'neuron_number_in_layer': ('neuroid', np.concatenate(
                        [np.arange(values.shape[-1]) for values in layer_representations.values()])),
                },
                dims=['presentation', 'neuroid'])
            neuroid_id = representations['layer'] + '--' + representations['neuron_number_in_layer'].values.astype(str)
            representations['neuroid_id'] = 'neuroid', neuroid_id
            output['neural'] = representations
        return output

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
