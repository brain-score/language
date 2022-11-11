import functools
import logging
from collections import OrderedDict
import re
from typing import Union, List, Tuple, Dict, Callable

import numpy as np
import torch
import xarray as xr
from numpy.core import defchararray
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput

from brainio.assemblies import DataAssembly, NeuroidAssembly, BehavioralAssembly
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils import fullname


class HuggingfaceSubject(ArtificialSubject):
    def __init__(
            self,
            model_id: str,
            region_layer_mapping: dict,
            model=None,
            tokenizer=None,
            task_heads: Union[None, Dict[ArtificialSubject.Task, Callable]] = None,
    ):
        """
            :param model_id: the model id i.e. name
            :param region_layer_mapping: commit which layers correspond to which regions.
                This can be left empty, but the model will not be able to be tested on neural benchmarks
            :param model: the model to run inference from. Using `AutoModelForCausalLM.from_pretrained` if `None`.
            :param tokenizer: the model's associated tokenizer. Using `AutoTokenizer.from_pretrained` if `None`.
            :param task_heads: a mapping from one or multiple tasks
                (:class:`~brainscore_language.artificial_subject.ArtificialSubject.Task`) to a function outputting the
                requested task output, given the basemodel's base output
                (:class:`~transformers.modeling_outputs.CausalLMOutput`).
        """
        self._logger = logging.getLogger(fullname(self))
        self.model_id = model_id
        self.region_layer_mapping = region_layer_mapping
        self.basemodel = (model if model is not None else AutoModelForCausalLM.from_pretrained(self.model_id))
        self.basemodel.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(self.model_id,
                                                                                               truncation_side='left')
        self.current_tokens = None  # keep track of current tokens

        self.neural_recordings: List[Tuple] = []  # list of `(recording_target, recording_type)` tuples to record
        self.behavioral_task: Union[None, ArtificialSubject.Task] = None
        task_mapping_default = {
            ArtificialSubject.Task.next_word: self.predict_next_word,
            ArtificialSubject.Task.reading_times: self.estimate_reading_times,
        }
        self.task_function_mapping_dict = {**task_mapping_default, **task_heads} if task_heads else task_mapping_default

    def identifier(self):
        return self.model_id

    def start_behavioral_task(self, task: ArtificialSubject.Task):
        self.behavioral_task = task
        self.output_to_behavior = self.task_function_mapping_dict[task]

    def start_neural_recording(self,
                               recording_target: ArtificialSubject.RecordingTarget,
                               recording_type: ArtificialSubject.RecordingType):
        self.neural_recordings.append((recording_target, recording_type))

    def digest_text(self, text: Union[str, List[str]]) -> Dict[str, DataAssembly]:
        """
        :param text: the text to be used for inference e.g. "the quick brown fox"
        :return: assembly of either behavioral output or internal neural representations
        """

        if type(text) == str:
            text = [text]

        output = {'behavior': [], 'neural': []}
        number_of_tokens = 0

        text_iterator = tqdm(text, desc='digest text') if len(text) > 100 else text  # show progress bar if many parts
        for part_number, text_part in enumerate(text_iterator):
            # prepare string representation of context
            context = self._prepare_context(text[:part_number + 1])
            context_tokens, number_of_tokens = self._tokenize(context, number_of_tokens)

            # prepare recording hooks
            hooks, layer_representations = self._setup_hooks()

            # run and remove hooks
            with torch.no_grad():
                base_output = self.basemodel(**context_tokens)
            for hook in hooks:
                hook.remove()

            # format output
            stimuli_coords = {
                'stimulus': ('presentation', [text_part]),
                'context': ('presentation', [context]),
                'part_number': ('presentation', [part_number]),
            }
            if self.behavioral_task:
                behavioral_output = self.output_to_behavior(base_output=base_output)
                behavior = BehavioralAssembly(
                    [behavioral_output],
                    coords=stimuli_coords,
                    dims=['presentation']
                )
                output['behavior'].append(behavior)
            if self.neural_recordings:
                representations = self.output_to_representations(layer_representations, stimuli_coords=stimuli_coords)
                output['neural'].append(representations)

        # merge over text parts
        self._logger.debug("Merging outputs")
        output['behavior'] = xr.concat(output['behavior'], dim='presentation').sortby('part_number') \
            if output['behavior'] else None
        output['neural'] = xr.concat(output['neural'], dim='presentation').sortby('part_number') \
            if output['neural'] else None
        return output

    def _prepare_context(self, context_parts):
        """
        Prepare a single string representation of a (possibly partial) input context
        for the model.
        """
        context = ' '.join(context_parts)

        # Remove erroneous spaces before punctuation.
        context = re.sub(r'\s+([.,!?;:])', r'\1', context)

        return context

    def _tokenize(self, context, num_previous_context_tokens):
        """
        Tokenizes the context, keeping track of the newly added tokens in `self.current_tokens`
        """
        context_tokens = self.tokenizer(context, truncation=True, return_tensors="pt")
        context_tokens.to('cuda' if torch.cuda.is_available() else 'cpu')
        # keep track of tokens in current `text_part`
        overflowing_encoding: list = np.array(context_tokens.encodings).item().overflowing
        num_overflowing = 0 if not overflowing_encoding else sum(len(overflow) for overflow in overflowing_encoding)
        self.current_tokens = {key: value[..., num_previous_context_tokens - num_overflowing:]
                               for key, value in context_tokens.items()}
        num_new_context_tokens = context_tokens['input_ids'].shape[-1] + num_overflowing
        return context_tokens, num_new_context_tokens

    def _setup_hooks(self):
        """ set up the hooks for recording internal neural activity from the model (aka layer activations) """
        hooks = []
        layer_representations = OrderedDict()
        for (recording_target, recording_type) in self.neural_recordings:
            layer_name = self.region_layer_mapping[recording_target]
            layer = self._get_layer(layer_name)
            hook = self._register_hook(layer, key=(recording_target, recording_type, layer_name),
                                       target_dict=layer_representations)
            hooks.append(hook)
        return hooks, layer_representations

    def output_to_representations(self, layer_representations: Dict[Tuple[str, str, str], np.ndarray], stimuli_coords):
        representation_values = np.concatenate([
            # Choose to use last token (-1) of values[batch, token, unit] to represent passage.
            values[:, -1:, :].squeeze(0) for values in layer_representations.values()],
            axis=-1)  # concatenate along neuron axis
        neuroid_coords = {
            'layer': ('neuroid', np.concatenate([[layer] * values.shape[-1]
                                                 for (recording_target, recording_type, layer), values
                                                 in layer_representations.items()])),
            'region': ('neuroid', np.concatenate([[recording_target] * values.shape[-1]
                                                  for (recording_target, recording_type, layer), values
                                                  in layer_representations.items()])),
            'recording_type': ('neuroid', np.concatenate([[recording_type] * values.shape[-1]
                                                          for (recording_target, recording_type, layer), values
                                                          in layer_representations.items()])),
            'neuron_number_in_layer': ('neuroid', np.concatenate(
                [np.arange(values.shape[-1]) for values in layer_representations.values()])),
        }
        neuroid_coords['neuroid_id'] = 'neuroid', functools.reduce(defchararray.add, [
            neuroid_coords['layer'][1], '--', neuroid_coords['neuron_number_in_layer'][1].astype(str)])
        representations = NeuroidAssembly(
            representation_values,
            coords={**stimuli_coords, **neuroid_coords},
            dims=['presentation', 'neuroid'])
        return representations

    def estimate_reading_times(self, base_output: CausalLMOutput):
        """
        :param base_output: the neural network's output
        :return: surprisal (in bits) as a proxy for reading times, following Smith & Levy 2013
            (https://www.sciencedirect.com/science/article/pii/S0010027713000413)
        """
        import torch.nn.functional as F
        # `base_output.logits` is (batch_size, sequence_length, vocab_size)
        logits = base_output.logits.squeeze(dim=0)

        FIRST_TOKEN_READING_TIME = np.nan
        if logits.shape[0] <= 1:  # sequence_length indicates we have only seen single word
            # since we have no context, we can't predict the next word
            # attempt to resolve by assuming a fixed reading time for the first word
            return FIRST_TOKEN_READING_TIME

        # get expectation (logits), shifted left by 1
        predicted_logits = logits[-self.current_tokens['input_ids'].shape[-1] - 1: - 1, :].contiguous()
        actual_tokens = self.current_tokens['input_ids'].squeeze(dim=0).contiguous()
        if actual_tokens.shape[0] == predicted_logits.shape[0] + 1:  # multiple tokens for first model input
            actual_tokens = actual_tokens[1:]  # we have no prior context to predict the 0th token

        # assume that reading time is additive, i.e. reading time of multiple tokens is
        # the sum of the surprisals of each individual token.
        # Note that this implementation similarly sums over the surprisal of multiple words,
        # e.g. for the surprisal of an entire sentence.
        cross_entropy = F.cross_entropy(predicted_logits, actual_tokens, reduction='sum') / np.log(2)
        return cross_entropy.to('cpu')

    def predict_next_word(self, base_output: CausalLMOutput):
        """
        :param base_output: the neural network's output
        :return: predicted next word
        """

        logits = base_output.logits
        pred_id = torch.argmax(logits, axis=2).squeeze()
        # Note that this is currently only predicting the next *token* which might not always be entire words.
        last_model_token_inference = pred_id[-1].tolist()
        next_word = self.tokenizer.decode(last_model_token_inference)
        # `next_word` often includes a space ` ` in front of the actual word. Since the task already tells us to output
        # a word, we can strip the space.
        # Note that the model might also predict a token completing the current word, e.g. `fox` -> `es` (`foxes`),
        # this is not caught in the current implementation.
        next_word = next_word.strip()
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
                       key: Tuple[str, str, str],
                       target_dict: dict) -> RemovableHandle:
        # instantiate parameters to function defaults; otherwise they would change on next function call
        def hook_function(_layer: torch.nn.Module, _input, output: torch.Tensor, key=key):
            if len(output) > 1:
                # fix for when taking out only the hidden state, thisis different from droput because of residual state
                # see :  https://github.com/huggingface/transformers/blob/c06d55564740ebdaaf866ffbbbabf8843b34df4b/src/transformers/models/gpt2/modeling_gpt2.py#L428
                target_dict[key] = self._tensor_to_numpy(output[0])
            else:
                target_dict[key] = self._tensor_to_numpy(output)

        hook = layer.register_forward_hook(hook_function)
        return hook

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().data.numpy()
