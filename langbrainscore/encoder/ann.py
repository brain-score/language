from enum import unique
import typing
import numpy as np
from langbrainscore.interface.encoder import _ANNEncoder
from collections import defaultdict
import string
import torch
from tqdm import tqdm
import xarray as xr


## Functions (probably to migrate later on)

def flatten_activations_per_sample(activations: dict):
    """
    Convert activations into dataframe format
    
    Args:
        Input (dict): key = layer, value = array of emb_dim
        
    Returns:
        arr_flat (np.ndarray): 1D ndarray of flattened activations across all layers
        layers_arr (np.ndarray): 1D ndarray of layer indices, corresponding to arr_flat
    """
    layers_arr = []
    arr_flat = []
    for layer, arr in activations.items(): # Iterate over layers
        arr = np.array(arr)
        arr_flat.append(arr)
        for i in range(arr.shape[0]):  # Iterate across units
            layers_arr.append(layer)
    arr_flat = np.concatenate(arr_flat, axis=0)  # concatenated activations across layers

    return arr_flat, np.array(layers_arr)


def aggregate_layers(hidden_states: dict,
                     **kwargs):
    """Input a hidden states dictionary (key = layer, value = 2D array of n_tokens x emb_dim)

    Args:
        hidden_states (dict): key = layer (int), value = 2D PyTorch tensor of shape (n_tokens, emb_dim)

    Raises:
        NotImplementedError

    Returns:
        dict: key = layer, value = array of emb_dim
    """
    emb_aggregation = kwargs.get('emb_aggregation')
    states_layers = dict()
    
    # Iterate over layers
    for i in hidden_states.keys():
        if emb_aggregation == 'last':
            state = hidden_states[i][-1, :]  # get last token
        elif emb_aggregation == 'mean':
            state = torch.mean(hidden_states[i], dim=0)  # mean over tokens
        elif emb_aggregation == 'median':
            state = torch.median(hidden_states[i], dim=0)  # median over tokens
        elif emb_aggregation == 'sum':
            state = torch.sum(hidden_states[i], dim=0)  # sum over tokens
        elif emb_aggregation == 'all' or emb_aggregation == None:
            state = hidden_states
        else:
            raise NotImplementedError('Sentence embedding method not implemented')

        states_layers[i] = state.detach().numpy()

    return states_layers


class HuggingFaceEncoder(_ANNEncoder):
    _pretrained_model_name_or_path = None

    def __init__(self, pretrained_model_name_or_path) -> None:
        self._pretrained_model_name_or_path = pretrained_model_name_or_path

        from transformers import AutoModel, AutoConfig, AutoTokenizer
        self.config = AutoConfig.from_pretrained(self._pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self._pretrained_model_name_or_path)
        self.model = AutoModel.from_pretrained(self._pretrained_model_name_or_path, config=self.config)
    
    def _case(self, sample: str = None,
              emb_case: typing.Union[str, None] = None):
        if emb_case == 'lower':
            sample = sample.lower()
        elif emb_case == 'upper':
            sample = sample.upper()
        else:
            sample = sample
            
        return sample
    
    
    def encode(self, dataset: 'langbrainscore.dataset.Dataset',
               context_dimension: str = None,
               bidirectional: bool = False,
               emb_case: typing.Union[str, None] = 'lower',
               emb_aggregation: typing.Union[str, None, typing.Callable] = 'last',
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

        stimuli = dataset.stimuli.values
        
        # Initialize the context group coordinate (obtain embeddings with context)
        if context_dimension is None:
            context_groups = np.arange(0, len(stimuli), 1)
        else:
            context_groups = dataset.stimuli.coords[context_dimension].values

        # Initialize list for storing activations for each stimulus with all layers flattened: flattened_activations
        # Initialize list for storing a list with layer ids ([0 0 0 0 ... 1 1 1 ...]) indicating which layers each neuroid came from
        flattened_activations, layer_ids = [], []
        
        ###############################################################################
        # ALL SAMPLES LOOP
        ###############################################################################
        _, unique_ixs = np.unique(context_groups, return_index=True) # Make sure context group order is preserved
        for group in tqdm(context_groups[np.sort(unique_ixs)]):
            mask_context = context_groups == group
            stimuli_in_context = stimuli[mask_context]  # Mask based on the context group

            # We want to tokenize all stimuli of this context group individually first in order to keep track of
            # which tokenized subunit belongs to what stimulus
            tokenized_stim_start_index = 0 # Store the index at which current stimulus starts (the past context ENDS) in the tokenized sequence
            
            states_sentences_across_stimuli = [] # Store states for each sample in this context group
            
            ###############################################################################
            # CONTEXT LOOP
            ###############################################################################
            for i, stimulus in enumerate(stimuli_in_context):
                # Mask based on the uni/bi-directional nature of models
                if not bidirectional:
                    stimuli_directional = stimuli_in_context[:i + 1]
                else:
                    stimuli_directional = stimuli_in_context

                stimuli_directional = ' '.join(stimuli_directional)
                stimuli_directional = self._case(sample=stimuli_directional, emb_case=emb_case)

                special_token_ids = self.tokenizer(' '.join(self.tokenizer.special_tokens_map.values()))['input_ids']
                special_token_ids = set(special_token_ids)

                # Tokenize the current stimulus only to get its length, and disable adding special tokens
                tokenized_current_stimulus = self.tokenizer(stimulus, padding=False, return_tensors='pt', add_special_tokens=False)
                tokenized_current_stim_length = tokenized_current_stimulus.input_ids.shape[1]
                tokenized_directional_context = self.tokenizer(stimuli_directional, padding=False, return_tensors='pt')

                # Get the hidden states
                result_model = self.model(tokenized_directional_context.input_ids, output_hidden_states=True, return_dict=True)
                hidden_states = result_model['hidden_states']  # dict with key=layer, value=3D tensor of dims: [batch, tokens, emb size]

                layer_wise_activations = dict()
                
                # Cut the 'irrelevant' context from the hidden states
                for idx_layer, layer in enumerate(hidden_states): # Iterate over layers
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
                states_sentences_agg = aggregate_layers(layer_wise_activations,
                                                        **{'emb_aggregation': emb_aggregation})
                # states_sentences is a dict with key = layer, value = array of emb dimension

                states_sentences_across_stimuli.append(states_sentences_agg)
                # states_sentences_across_stimuli store all the hidden states for the current context group across all stimuli

            ###############################################################################
            # END CONTEXT LOOP
            ###############################################################################

            # Flatten activations across layers and package as xarray
            flattened_activations_and_layer_ids = [*map(flatten_activations_per_sample,
                                                        states_sentences_across_stimuli)]
            for f_as, l_ids in flattened_activations_and_layer_ids:
                flattened_activations += [f_as]
                layer_ids += [l_ids]
                assert len(f_as) == len(l_ids) # Assert all layer lists are equal

        ###############################################################################
        # END ALL SAMPLES LOOP
        ###############################################################################
        
        encoded_data = np.expand_dims(np.vstack(flattened_activations), axis=2)

        # Generate xarray DataSet
        xr_encode = xr.DataArray(
            encoded_data,
            dims=("sampleid", "neuroid", "timeid"),
            coords={
                "sampleid": dataset._dataset.sampleid.values,
                "neuroid": np.arange(np.sum([len(states_sentences_agg[x]) for x in states_sentences_agg])),
                "timeid": np.arange(1),
                "layer": ('neuroid', np.array(layer_ids[0], dtype='int64')),
            }
        )


        # Add in sampleid coordinates from the original dataset
        for k in dataset._dataset.to_dataset(name='data').drop_dims(['neuroid', 'timeid']).coords: # Keeps only sampleid, and has no data
            xr_encode = xr_encode.assign_coords({k: ('sampleid', dataset._dataset[k].data)})

        return xr_encode

class PTEncoder(_ANNEncoder):
    def __init__(self, ptid) -> None:
        self._ptid = ptid

    def encode(self, dataset: 'langbrainscore.dataset.Dataset') -> xr.DataArray:
        #TODO
        pass
