import typing
from enum import unique

import numpy as np
import xarray as xr
from langbrainscore.dataset import Dataset
from langbrainscore.interface import _ModelEncoder
from langbrainscore.utils.encoder import *
from langbrainscore.utils.xarray import copy_metadata
from tqdm import tqdm


class HuggingFaceEncoder(_ModelEncoder):
    def __init__(self, model_id) -> "HuggingFaceEncoder":
        super().__init__(model_id)

        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.config = AutoConfig.from_pretrained(self._model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self.model = AutoModel.from_pretrained(self._model_id, config=self.config)

    def encode(
        self,
        dataset: Dataset,
        context_dimension: str = None,
        bidirectional: bool = False,
        emb_case: typing.Union[str, None] = "lower",
        emb_aggregation: typing.Union[str, None, typing.Callable] = "last",
        emb_norm: typing.Union[str, None] = None,
        emb_outlier: typing.Union[str, None] = None,
    ) -> xr.DataArray:
        """
        Input a langbrainscore Dataset and return a xarray DataArray of sentence embeddings given the specified
        parameters (in ANNEmbeddingConfig).

        Args:
            dataset (langbrainscore.dataset.DataSet): [description]
            context_dimension (str, optional): the name of a dimension in our xarray-like dataset objcet that provides
                                                groupings of sampleids (stimuli) that should be used
                                                as context when generating encoder representations. for instance, in a word-by-word
                                                presentation paradigm we (may) still want the full sentence as context. [default: None].
            bidirectional (bool, optional): if True, allows using "future" context to generate the representation for a current token
                                            otherwise, only uses what occurs in the "past". some might say, setting this to False
                                            gives you a more biologically plausibly analysis downstream (: [default: False]
            emb_case (str, optional): which casing to provide to the textual input that is fed into the encoder model. Defaults to 'lower'.
            emb_aggregation (typing.Union[str, None, typing.Callable], optional): how to aggregate the hidden states of the encoder
            emb_norm (typing.Union[str, None], optional): how to normalize the hidden states of the encoder. Defaults to None.
            emb_outlier (typing.Union[str, None], optional): how to remove outliers from the hidden states of the encoder. Defaults to None.

        Raises:
            NotImplementedError: [description]
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        self.model.eval()
        
        with_special_tokens = self.tokenizer("brainscore")['input_ids']
        first_token_id, *_ = self.tokenizer("brainscore", add_special_tokens=False)['input_ids']
        special_token_offset = with_special_tokens.index(first_token_id)

        stimuli = dataset.stimuli.values

        # Initialize the context group coordinate (obtain embeddings with context)
        context_groups = get_context_groups(dataset, context_dimension)

        # Initialize list for storing activations for each stimulus with all layers flattened: flattened_activations
        # Initialize list for storing a list with layer ids ([0 0 0 0 ... 1 1 1 ...]) indicating which layers each neuroid came from
        flattened_activations, layer_ids = [], []

        ###############################################################################
        # ALL SAMPLES LOOP
        ###############################################################################
        _, unique_ixs = np.unique(context_groups, return_index=True)
        # Make sure context group order is preserved
        for group in tqdm(context_groups[np.sort(unique_ixs)]):
            mask_context = context_groups == group
            stimuli_in_context = stimuli[mask_context]
            # Mask based on the context group

            
            # We want to tokenize all stimuli of this context group individually first in order to keep track of
            # which tokenized subunit belongs to what stimulus
            # Store the index at which current stimulus starts (the past context ENDS) in the tokenized sequence
            tokenized_stim_start_index = special_token_offset 

            states_sentences_across_stimuli = []
            # Store states for each sample in this context group

            ###############################################################################
            # CONTEXT LOOP
            ###############################################################################
            for i, stimulus in enumerate(stimuli_in_context):
                stimulus = set_case(sample=stimulus, emb_case=emb_case)

                # Mask based on the uni/bi-directional nature of models
                if not bidirectional:
                    stimuli_directional = stimuli_in_context[: i + 1]
                else:
                    stimuli_directional = stimuli_in_context

                stimuli_directional = " ".join(stimuli_directional)
                stimuli_directional = set_case(
                    sample=stimuli_directional, emb_case=emb_case
                )

                special_token_ids = self.tokenizer(
                    " ".join(self.tokenizer.special_tokens_map.values())
                )["input_ids"]
                special_token_ids = set(special_token_ids)

                # Tokenize the current stimulus only to get its length, and disable adding special tokens
                tokenized_current_stim = self.tokenizer(
                    stimulus,
                    padding=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                )  # todo double check this!
                tokenized_current_stim_length = tokenized_current_stim.input_ids.shape[
                    1
                ]
                tokenized_directional_context = self.tokenizer(
                    stimuli_directional, 
                    padding=False, return_tensors="pt", add_special_tokens=True,
                )

                # Get the hidden states
                result_model = self.model(
                    tokenized_directional_context.input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = result_model[
                    "hidden_states"
                ]  # dict with key=layer, value=3D tensor of dims: [batch, tokens, emb size]

                layer_wise_activations = dict()

                # Cut the 'irrelevant' context from the hidden states
                for idx_layer, layer in enumerate(hidden_states):  # Iterate over layers
                    layer_wise_activations[idx_layer] = layer[
                        # batch (singleton)
                        :,
                        # n_tokens
                        tokenized_stim_start_index: 
                            tokenized_stim_start_index + tokenized_current_stim_length,
                        # emb_dim (e.g., 768)
                        :,
                    ].squeeze()  # collapse batch dim to obtain shape (n_tokens, emb_dim)

                tokenized_stim_start_index += tokenized_current_stim_length

                # Aggregate hidden states within a sample
                states_sentences_agg = aggregate_layers(
                    layer_wise_activations, **{"emb_aggregation": emb_aggregation}
                )
                # states_sentences is a dict with key = layer, value = array of emb dimension

                states_sentences_across_stimuli.append(states_sentences_agg)
                # states_sentences_across_stimuli store all the hidden states for the current context group across all stimuli

            ###############################################################################
            # END CONTEXT LOOP
            ###############################################################################

            # Flatten activations across layers and package as xarray
            flattened_activations_and_layer_ids = [
                *map(flatten_activations_per_sample, states_sentences_across_stimuli)
            ]
            for f_as, l_ids in flattened_activations_and_layer_ids:
                flattened_activations += [f_as]
                layer_ids += [l_ids]
                assert len(f_as) == len(l_ids)  # Assert all layer lists are equal

        ###############################################################################
        # END ALL SAMPLES LOOP
        ###############################################################################

        encoded_dataset = copy_metadata(
            repackage_flattened_activations(
                flattened_activations, states_sentences_agg, layer_ids, dataset
            ),
            dataset.contents,
            "sampleid",
        )

        return encoded_dataset


class PTEncoder(_ModelEncoder):
    def __init__(self, model_id: str) -> "PTEncoder":
        super().__init__(model_id)

    def encode(self, dataset: "langbrainscore.dataset.Dataset") -> xr.DataArray:
        # TODO
        pass
