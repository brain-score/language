
from enum import unique
import typing
import IPython
import numpy as np
from langbrainscore.interface.encoder import _ANNEncoder
from collections import defaultdict
import string
import torch


## Functions (probably to migrate later on)

def flatten_activations(activations):
    """
    Convert activations into dataframe format
    Input: dict, key = layer, item = 2D array of stimuli x units
    Output: pd dataframe, stimulus ID x MultiIndex (layer, unit)
    """
    labels = []
    arr_flat = []
    for layer, arr in activations.items():
        arr = np.array(arr) # for each layer
        arr_flat.append(arr)
        for i in range(arr.shape[1]): # across units
            labels.append((layer, i))
    arr_flat = np.concatenate(arr_flat, axis=1) # concatenated activations across layers
    df = pd.DataFrame(arr_flat)
    df.columns = pd.MultiIndex.from_tuples(labels) # rows: stimuli, columns: units
    return df

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


class ANNEmbeddingConfig:
    def __init__(self, aggregation: typing.Union[str, None, typing.Callable] = 'last',
                 norm=None, outlier_removal=None) -> None:

        if type(aggregation) in (str, type(None)) and aggregation not in {'first', 'last', 'mean', 'median', 'all', None}:
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
            xarray Dataset: an xarray Dataset with coordinates as follows: 'sample[id]', 'neuroid', 'time[id]', 
                            containing an encoder representation per sample
        """
        self.model.eval() 
        # only for gpt? check the general case 
        n_layer = self.model.config.n_layer
        states_sentences = defaultdict(list)

        stimuli = dataset.stimuli.values
        if context_dimension is None:
            context_groups = np.arange(0, len(stimuli), 1)
        else:
            context_groups = dataset.coords[context_dimension].values
        
        for group in np.unique(context_groups):
            mask = context_groups == group
            stimuli_in_context = stimuli[mask]

            # we want to tokenize all stimuli of this ctxt group individually first in order to keep track of
            # which tokenized subunit belongs to what stimulus

            tokenized_lengths = []
            word_ids_by_stim = [] # contains a list of token->word_id lists per stimulus
            for stimulus in stimuli_in_context:
                tokenized_stim = self.tokenizer(stimulus, padding=False, truncate=False)
                tokenized_lengths += [len(tokenized_stim)]
                
                word_ids = tokenized_stim.word_ids() # make sure this is positional, not based on word identity        
                word_ids_by_stim += [word_ids]

            # we want to concatenate all stim of this ctxt group and obtain an encoder representation 

            # 
            concatenated_chunk = ' '.join(stimuli_in_context)
            tokenized_chunk = self.tokenizer(concatenated_chunk)
            if len(tokenized_chunk) >= 512: # too many tokens
                raise ValueError(f'context too long at {len(tokenized_chunk)} tokens') 

            # in this case, the concatenated chunk should include all the stimuli (even the future ones)
            # in the context group
            if bidirectional:


            # this will be a matrix with (n_stimuli_in_context, max_n_tokens, n_dims)
            contextual_rep
            

        # stimulus is the generalization of a "sentence" to words as well as sentences
        for count, stimulus in enumerate(dataset.stimuli.values):

            if case == 'lower':
                stimulus = stimulus.lower()
            elif case == 'upper':
                stimulus = stimulus.upper()

            tokenized: 'tokenizers.BatchEncoding' = self.tokenizer(sentence)
            word_ids = tokenized.word_ids() # make sure this is positional, not based on word identity        
            input_ids = tokenized['input_ids']

            result_model = self.model(input_ids, output_hidden_states=True, return_dict=True)
            hidden_states = result_model['hidden_states'] # 3D tensor of dims: [batch, tokens, emb size]


            def aggregate_layers(hidden_states, aggregation_args):
                """[summary]

                Args:
                    hidden_states (torch.Tensor): pytorch tensor of shape (batch_size, n_items, dims)
                    aggregation_args (ANNEmbeddingConfig): an object specifying the method to use for aggregating
                                                            representations across items within a layer

                Raises:
                    NotImplementedError

                Returns:
                    np.ndarray: the aggregated array
                """
                method = aggregation_args.aggregation
                for i in range(n_layer+1): # for each layer
                    if method == 'last':
                        state = hidden_states[i].squeeze()[-1, :] # get last token
                    elif method == 'mean':
                        state = torch.mean(hidden_states[i].squeeze(), dim=0) # mean over tokens
                    elif method == 'median':
                        state = torch.median(hidden_states[i].squeeze(), dim=0) # median over tokens
                    elif method == 'sum':
                        state = torch.sum(hidden_states[i].squeeze(), dim=0) # sum over tokens
                    elif method == 'all' or method == None:
                        state = hidden_states
                    else:
                        raise NotImplementedError('Sentence embedding method not implemented')
                    
                    return state.detach().numpy()


            #### first, we do word-level aggregation according to word_level_args
            # so, we iterate through groups of tokens for each word and then aggregate per word
            # we want to enforce that there is always _some_ sort of aggregation at the word level
            # so we raise an exception if anything other than a method (i,e, 'all', or None) is passed

            if word_level_args.aggregation in (None, 'all'):
                raise ValueError('must aggregate at the word level! try "last" or "mean"')

            word_level_hidden_states = []            

            unique_words_position_ids = sorted(set(word_ids)) # in order of appearance
            for word_id in unique_words_position_ids:
                mask = word_ids == word_id
                # this_aggregated will have shape (batch=1, n_tokens=1, dims), i.e., (1,1,dims)
                this_aggregated = aggregate_layers(hidden_states[:, mask, :], word_level_args)
                word_level_hidden_states += [this_aggregated]   

            b, _, d = hidden_states.shape
            word_level_hidden_states = np.ndarray(word_level_hidden_states).reshape(b, -1, d)
            _, num_words, _ = word_level_hidden_states.shape
            assert num_words == len(unique_words_position_ids), 'Mismatch in the number of words before and after aggregation :('

            # we expect this to have dimensions (b=1, num_words, d) or (b=1, 1, d)
            sentence_level_hidden_states = aggregate_layers(word_level_hidden_states, sentence_level_args).squeeze()


            states_sentences[i].append(state)
            #### first, we do word-level aggregation according to word_level_args





class PTEncoder(_ANNEncoder):
    def __init__(self, ptid) -> None:
        super().__init__(self)
        self._ptid = ptid

    def encode(self, dataset: 'langbrainscore.dataset.DataSet') -> None:
        pass
