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
    Input: dict, key = layer, item = 2D array of stimuli x units
    Output: pd dataframe, stimulus ID x MultiIndex (layer, unit)
    """
    labels = []
    layers_arr = []
    arr_flat = []
    for layer, arr in activations.items():
        arr = np.array(arr)  # for each layer
        arr_flat.append(arr)
        for i in range(arr.shape[0]):  # across units
            # labels.append((layer, i))
            layers_arr.append(layer)
    arr_flat = np.concatenate(arr_flat, axis=0)  # concatenated activations across layers
    # df.columns = pd.MultiIndex.from_tuples(labels)  # rows: stimuli, columns: units
    return arr_flat, np.array(layers_arr)

#def flatten_activations(activations: list, layer_ids: list):
#	return np.concatenate(activations), np.concatenate(layer_ids)


def get_activations(model, tokenizer, stimuli, emb_method='last-tok',
                    case='lower', punc='strip-all', punc_exceptions=[],
                    norm=None, outlier_removal=None,
                    verbose=True):
    """
    Obtain activations from (HF) models.

    :param model: model object, HF
    :paral tokenizer: tokenizer object, HF
    :param stimuli: list/array, [to do, edit] containing strings
    :param emb_method: str, denoting how to obtain sentence embedding
    :param case: str, denoting which casing to use
    :param punc: str, denoting which operations to perform regarding punctuation
    :param punc_exceptions: list, denoting which punctuation to NOT strip if punc == 'strip-all'
    :param norm: str, denoting how to normalize the embeddings
    :param outlier_removel: str, denoting whether to remove 'outliers' from the embedding

    Returns dict with key = layer, item = 2D array of stimuli x units
    """
    
    return states_sentences


def aggregate_layers(hidden_states: dict, aggregation_args):
    """[summary]

    Args:
        hidden_states (torch.Tensor): pytorch tensor of shape (n_items, dims)
        aggregation_args (ANNEmbeddingConfig): an object specifying the method to use for aggregating
                                                representations across items within a layer

    Raises:
        NotImplementedError

    Returns:
        np.ndarray: the aggregated array
    """
    method = aggregation_args.aggregation
    states_layers = dict()
    # n_layers = len(hidden_states)
    for i in hidden_states.keys():  # for each layer
        if method == 'last':
            state = hidden_states[i][-1, :]  # get last token
        elif method == 'mean':
            state = torch.mean(hidden_states[i], dim=0)  # mean over tokens
        elif method == 'median':
            state = torch.median(hidden_states[i], dim=0)  # median over tokens
        elif method == 'sum':
            state = torch.sum(hidden_states[i], dim=0)  # sum over tokens
        elif method == 'all' or method == None:
            state = hidden_states
        else:
            raise NotImplementedError('Sentence embedding method not implemented')
        
        states_layers[i] = state.detach().numpy()
        
    return states_layers


class ANNEmbeddingConfig:
    def __init__(self, aggregation: typing.Union[str, None, typing.Callable] = 'last',
                 norm=None, outlier_removal=None) -> None:
        if type(aggregation) in (str, type(None)) and aggregation not in {'first', 'last', 'mean', 'median', 'all',
                                                                          None}:
            raise ValueError(f'aggregation type {aggregation} not supported')
        # else: it could be a Callable (user implementing their own aggregation method); we won't sanity-check that here
        self.aggregation = aggregation
        self.norm = norm
        self.outlier_removal = outlier_removal


class HuggingFaceEncoder(_ANNEncoder):
    _pretrained_model_name_or_path = None
    
    def __init__(self, pretrained_model_name_or_path) -> None:
        # super().__init__(self)
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        
        from transformers import AutoModel, AutoConfig, AutoTokenizer
        self.config = AutoConfig.from_pretrained(self._pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self._pretrained_model_name_or_path)
        self.model = AutoModel.from_pretrained(self._pretrained_model_name_or_path, config=self.config)
    
    def encode(self, dataset: 'langbrainscore.dataset.DataSet',
               word_level_args: ANNEmbeddingConfig = ANNEmbeddingConfig(aggregation='last'),
               sentence_level_args: ANNEmbeddingConfig = ANNEmbeddingConfig(aggregation='last'),
               case: str = 'lower', punct: typing.Union[str, None, typing.List[str]] = None,
               context_dimension: str = None, bidirectional: bool = False):
        """[summary]

        Args:
            dataset (langbrainscore.dataset.DataSet): [description]
            word_level_args (ANNEmbeddingConfig, optional): [description]. Defaults to ANNEmbeddingConfig(aggregation='last').
            sentence_level_args (ANNEmbeddingConfig, optional): [description]. Defaults to ANNEmbeddingConfig(aggregation='last').
            case (str, optional): [description]. Defaults to 'lower'.
            punct (typing.Union[str, None, typing.List[str]], optional): [description]. Defaults to None.
            context_dimension (str, optional): the name of a dimension in our xarray-like dataset objcet that provides
                                                groupings of sampleids (stimuli) that should be used
                                                as context when generating encoder representations. for instance, in a word-by-word
                                                presentation paradigm we (may) still want the full sentence as context. [default: None].
            bidirectional (bool, optional): if True, allows using "future" context to generate the representation for a current token
                                            otherwise, only uses what occurs in the "past". some might say, setting this to False
                                            gives you a more biologically plausibly analysis downstream (: [default: False]

        Raises:
            NotImplementedError: [description]
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        self.model.eval()
        # n_layer = self.model.config.n_layer
        
        stimuli = dataset.stimuli.values
        if context_dimension is None:
            context_groups = np.arange(0, len(stimuli), 1)
        else:
            context_groups = dataset.stimuli.coords[context_dimension].values
        
        # stores activations for each sitmulus as all layers flattened
        # layer_ids stores a list similar to [0 0 0 0 ... 1 1 1 ...] indicating which layers
        # each neuroid/dimension came from
        flattened_activations, layer_ids = [], []
        ###############################################################################                  
        # ALL SAMPLES LOOP
        ###############################################################################                  
        # NOTE: np.uniques does NOT preserve the order of first appearance of an item
        # in its return value. it returns a sorted collection. so we have to use 
        # return_index
        # https://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
        _, unique_ixs = np.unique(context_groups, return_index=True)
        for group in tqdm(context_groups[np.sort(unique_ixs)]):
            mask_context = context_groups == group
            stimuli_in_context = stimuli[mask_context]  # mask based on the context group
            
            # we want to tokenize all stimuli of this ctxt group individually first in order to keep track of
            # which tokenized subunit belongs to what stimulus
            # this stores the index at which current stimulus starts (the past context ENDS)
            # in the tokenized sequence
            tokenized_stim_start_index = 0
            word_ids_by_stim = []  # contains a list of token->word_id lists per stimulus
            states_sentences_across_stimuli = []
            ###############################################################################                  
            # CONTEXT LOOP
            ###############################################################################                  
            for i, stimulus in enumerate(stimuli_in_context):
                # print(f'encoding stimulus {i} of {len(stimuli_in_context)}')
                # mask based on the uni/bi-directional nature of models :)
                if not bidirectional:
                    stimuli_directional = stimuli_in_context[:i + 1]
                else:
                    stimuli_directional = stimuli_in_context
                
                stimuli_directional = ' '.join(stimuli_directional)
                
                if case == 'lower':
                    stimuli_directional = stimuli_directional.lower()
                elif case == 'upper':
                    stimuli_directional = stimuli_directional.upper()
                else:
                    stimuli_directional = stimuli_directional
                
                # tokenize the stimuli
                #                                > maybe here instead of 'stimuli_directional' we 
                #                                   should use stimuli_in_context[:i+1] for both
                #                                   unidirectional and bidirectional cases?

                # we tokenize the current stimulus only to get its length, and thus, we disable adding special tokens
                tokenized_current_stimulus = self.tokenizer(stimulus, padding=False, return_tensors='pt', add_special_tokens=False)
                tokenized_current_stim_length = tokenized_current_stimulus.input_ids.shape[1]
                tokenized_directional_context = self.tokenizer(stimuli_directional, padding=False, return_tensors='pt')
                
                # Get the hidden states
                result_model = self.model(tokenized_directional_context.input_ids, output_hidden_states=True, return_dict=True)
                hidden_states = result_model['hidden_states']  # dict with key=layer, value=3D tensor of dims: [batch, tokens, emb size]
                
                # word_ids = tokenized_stim.word_ids()  # make sure this is positional, not based on word identity
                # word_ids_by_stim += [word_ids]  # contains the ids for each tokenized words in stimulus todo: maybe get rid of this?
                
                layer_wise_activations = dict()   
                # now cut the 'irrelevant' context from the hidden states
                for idx_layer, layer in enumerate(hidden_states):
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

                # aggregate within  a stimulus
                states_sentences_agg = aggregate_layers(layer_wise_activations,
                                                        aggregation_args=sentence_level_args)  # fix todo to aggregation args,
                # dict with key=layer, value=array of # size [emb dim]
                
                states_sentences_across_stimuli.append(states_sentences_agg)
                # now we have all the hidden states for the current context group across all stimuli,
                # emb_dim = states_sentences_across_stimuli[0][0].shape[-1]

            ###############################################################################                  
            # END CONTEXT LOOP
            ###############################################################################                  

            # flatten across layers and package as xarray
            flattened_activations_and_layer_ids = [*map(flatten_activations_per_sample, 
                                                        states_sentences_across_stimuli)]
            for f_as, l_ids in flattened_activations_and_layer_ids:
                flattened_activations += [f_as]
                layer_ids += [l_ids]
            # assert all layer lists are equal

        ###############################################################################                  
        # END ALL SAMPLES LOOP
        ###############################################################################                  

    
        #np.vstack(flattened_activations)
        encoded_data = np.expand_dims(np.vstack(flattened_activations), axis=2)

        # generate xarray DataSet
        xr_encode = xr.DataArray(
            encoded_data,
            dims=("sampleid", "neuroid", "timeid"),
            coords={
                "sampleid": dataset._dataset.sampleid.values,
                # the below way of specifying neuroids assumes all layers have the same # dimensions as the last layer
                "neuroid": np.arange(np.sum([len(states_sentences_agg[x]) for x in states_sentences_agg])),  # check
                "timeid": np.arange(1),
                "layer": ('neuroid', np.array(layer_ids[0], dtype='int64')),
            }
        ) 


        # TODO: explain what this does extensively
        for k in dataset._dataset.to_dataset(name='data').drop_dims(['neuroid', 'timeid']).coords: #<- keeps only sampleid, and has no data
            xr_encode = xr_encode.assign_coords({k: ('sampleid', dataset._dataset[k].data)})

        return xr_encode

class PTEncoder(_ANNEncoder):
    def __init__(self, ptid) -> None:
        super().__init__(self)
        self._ptid = ptid
    
    def encode(self, dataset: 'langbrainscore.dataset.DataSet') -> None:
        pass
