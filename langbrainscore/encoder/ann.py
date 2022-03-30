import typing
from enum import unique

import numpy as np
import xarray as xr
from langbrainscore.dataset import Dataset
from langbrainscore.interface import _ModelEncoder
from langbrainscore.utils.encoder import (
        set_case, aggregate_layers, 
        flatten_activations_per_sample,
        repackage_flattened_activations,
        get_context_groups, get_torch_device,
        preprocess_activations
    )
from langbrainscore.utils.xarray import copy_metadata
from tqdm import tqdm



class HuggingFaceEncoder(_ModelEncoder):
    def __init__(self, model_id, device=None) -> "HuggingFaceEncoder":
        super().__init__(model_id)

        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.device = device or get_torch_device()
        self.config = AutoConfig.from_pretrained(self._model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        try:
            self.model = AutoModel.from_pretrained(self._model_id, config=self.config).to(self.device)
        except RuntimeError:
            self.device = 'cpu'
            self.model = AutoModel.from_pretrained(self._model_id, config=self.config)

    
    def encode(
        self,
        dataset: Dataset,
        context_dimension: str = None,
        bidirectional: bool = False,
        emb_case: typing.Union[str, None] = "lower",
        emb_aggregation: typing.Union[str, None, typing.Callable] = "last",
        emb_preproc: typing.Union[list, np.ndarray] = ['demean'],
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
            emb_preproc (typing.Union[list, np.ndarray, None], optional): a list of preprocessing functions to apply to the embeddings.
                        Processing is done layer-wise.

        Raises:
            NotImplementedError: [description]
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        self.model.eval()
        special_token_offset = self.get_special_token_offset()
        stimuli = dataset.stimuli.values
        # Initialize the context group coordinate (obtain embeddings with context)
        context_groups = get_context_groups(dataset, context_dimension)

        # list for storing activations for each stimulus with all layers flattened 
        # list for storing layer ids ([0 0 0 0 ... 1 1 1 ...]) indicating which layer each 
        # neuroid (representation dimension) came from
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

            # to store the index at which current stimulus starts (the past context ENDS) in the tokenized sequence
            # so that we can extract each subsequent stimulus representations without the context representations
            tokenized_stim_start_index = special_token_offset

            # store model states for each stimulus in this context group
            states_sentences_across_stimuli = []

            ###############################################################################
            # CONTEXT LOOP
            ###############################################################################
            for i, stimulus in enumerate(stimuli_in_context):
                stimulus = set_case(sample=stimulus, emb_case=emb_case)

                # extract stim to encode based on the uni/bi-directional nature of models
                if not bidirectional:
                    stimuli_directional = stimuli_in_context[: i + 1]
                else:
                    stimuli_directional = stimuli_in_context

                stimuli_directional = " ".join(stimuli_directional)
                stimuli_directional = set_case(sample=stimuli_directional, emb_case=emb_case)

                # Tokenize the current stimulus only to get its length, and disable adding special tokens
                tokenized_current_stim = self.tokenizer(
                    stimulus,
                    padding=False, return_tensors="pt", add_special_tokens=False,
                ) 
                tokenized_current_stim_length = tokenized_current_stim.input_ids.shape[1]

                tokenized_directional_context = self.tokenizer(
                    stimuli_directional, 
                    padding=False, return_tensors="pt", add_special_tokens=True,
                ).to(self.device)

                # Get the hidden states
                result_model = self.model(
                    tokenized_directional_context.input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # dict with key=layer, value=3D tensor of dims: [batch, tokens, emb size]
                hidden_states = result_model["hidden_states"]  

                layer_wise_activations = dict()
                # Cut the "irrelevant" context from the hidden states
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
                    # ^ do we have to .detach() tensors here?

                # keep tracking lengths so we know where each next stimulus begins
                tokenized_stim_start_index += tokenized_current_stim_length

                # Aggregate hidden states within a sample
                # states_sentences_agg is a dict with key = layer, value = array of emb dimension
                states_sentences_agg = aggregate_layers(
                    layer_wise_activations, **{"emb_aggregation": emb_aggregation}
                )

                # states_sentences_across_stimuli store all the hidden states for the current context group across all stimuli
                states_sentences_across_stimuli.append(states_sentences_agg)

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
        
        # Stack flattened activations and layer ids to obtain [n_samples, emb_din * n_layers]
        activations_2d = np.vstack(flattened_activations)
        layer_ids_1d = np.squeeze(np.unique(np.vstack(layer_ids), axis=0))
        
        ###############################################################################
        # PREPROCESS ACTIVATIONS
        ###############################################################################
        if len(emb_preproc) > 0: # Preprocess activations
            activations_2d, layer_ids_1d = preprocess_activations(
                activations_2d=activations_2d,
                layer_ids_1d=layer_ids_1d,
                emb_preproc=emb_preproc,
            )
        
        assert(activations_2d.shape[1] == len(layer_ids_1d))
        assert(activations_2d.shape[0] == len(stimuli))
        
        # Package activations as xarray and reapply metadata
        encoded_dataset = copy_metadata(
            repackage_flattened_activations(
                activations_2d=activations_2d,
                layer_ids_1d=layer_ids_1d,
                dataset=dataset,
            ),
            dataset.contents,
            "sampleid",
        )

        return encoded_dataset


    def get_special_token_offset(self) -> int:
        '''
        the offset (no. of tokens in tokenized text) from the start to exclude
        when extracting the representation of a particular stimulus. this is
        needed when the stimulus is evaluated in a context group to achieve
        correct boundaries (otherwise we get off-by-context errors)
        '''
        with_special_tokens = self.tokenizer("brainscore")['input_ids']
        first_token_id, *_ = self.tokenizer("brainscore", add_special_tokens=False)['input_ids']
        special_token_offset = with_special_tokens.index(first_token_id)
        return special_token_offset


class PTEncoder(_ModelEncoder):
    def __init__(self, model_id: str) -> "PTEncoder":
        super().__init__(model_id)

    def encode(self, dataset: "langbrainscore.dataset.Dataset") -> xr.DataArray:
        # TODO
        pass
