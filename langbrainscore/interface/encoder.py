from __future__ import annotations
from abc import ABC, abstractmethod
import typing
import numpy as np
import pandas as pd

import langbrainscore

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
    
    model.eval() 
    n_layer = model.config.n_layer
    max_n_tokens = model.config.n_positions
    states_sentences = defaultdict(list)
    
    if verbose:
        print(f'Computing activations for {len(sents)} sentences')

    for count, sent in enumerate(stimuli):
        if case == 'lower': # todo deal with these nicely
            sent = sent.lower()
        if case == 'upper':
            sent = sent.upper()
            
        input_ids = torch.tensor(tokenizer.encode(sent))

        result_model = model(input_ids, output_hidden_states=True, return_dict=True)
        hidden_states = result_model['hidden_states'] # 3D tensor of dims: [batch, tokens, emb size]
        
        for i in range(n_layer+1): # for each layer
            if emb_method == 'last-tok':
                state = hidden_states[i].squeeze()[-1, :].detach().numpy() # get last token
            elif emb_method == 'mean-tok':
                state = torch.mean(hidden_states[i].squeeze(), dim=0).detach().numpy() # mean over tokens
            elif emb_method == 'median-tok':
                state = torch.median(hidden_states[i].squeeze(), dim=0).detach().numpy() # median over tokens
            elif emb_method == 'sum-tok':
                state = torch.sum(hidden_states[i].squeeze(), dim=0).detach().numpy() # sum over tokens
            elif emb_method == 'all-tok':
                raise NotImplementedError()
            else:
                print('Sentence embedding method not implemented')
                raise NotImplementedError()

            states_sentences[i].append(state)
            
    return states_sentences


class Encoder(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def encode(self, dataset: 'langbrainscore.dataset.DataSet') -> pd.DataFrame:
        return NotImplemented


class BrainEncoder(Encoder):
    '''
    This class provides a wrapper around real-world brain data of various kinds
        which may be: 
            - neuroimaging [fMRI, PET]
            - physiological [ERP, MEG, ECOG]
            - behavioral [RT, Eye-tracking]
    across several subjects. The class implements `BrainEncoder.encode` which takes in
    a collection of stimuli (typically `np.array` or `list`) 
    '''

    _dataset: langbrainscore.dataset.Dataset = None

    def __init__(self, dataset = None) -> None:
        # if not isinstance(dataset, langbrainscore.dataset.BrainDataset):
        #     raise TypeError(f"dataset must be of type `langbrainscore.dataset.BrainDataset`, not {type(dataset)}")
        self._dataset = dataset

    @property
    def dataset(self) -> langbrainscore.dataset.Dataset:
        return self._dataset

    # @typing.overload
    # def encode(self, stimuli: typing.Union[np.array, list]): ...
    def encode(self, dataset: 'langbrainscore.dataset.BrainDataSet' = None):
        """returns an "encoding" of stimuli (passed in as a BrainDataset)

        Args:
            stimuli (langbrainscore.dataset.BrainDataset):

        Returns:
            pd.DataFrame: neural recordings for each stimulus, multi-indexed 
                          by layer (trivially just 1 layer)
        """        
        
        dataset = dataset or self.dataset

        if (timeid_dims := dataset._dataset.dims['timeid']) > 1:
            return dataset._dataset.mean('timeid')
        elif timeid_dims == 1:
            return dataset._dataset.squeeze('timeid')
        else:
            raise ValueError(f'timeid has invalid shape {timeid_dims}')



class ANNEncoder(Encoder):
    def __init__(self) -> None:
        super().__init__(self)
        pass


    def encode(self, dataset: 'langbrainscore.dataset.DataSet'):
        """[summary]
        
        # Todo: Arguments: 
        Embedding method (emb_method): last-tok, mean-tok, median-tok, sum-tok, all-tok, 
        Casing (case): lower, upper, None (no edits)
        Punctuation (punc): strip-all, None
        Punctuation exceptions, i.e. what NOT to strip (punc_exceptions): default: []
        Standardization/normalization (norm): None, row, col
        Outlier removal (outlier_removal): None, 
        
        Args:
            stimuli (langbrainscore.dataset.DataSet): [description]

        Returns:
            pd.DataFrame: neural recordings for each stimulus, multi-indexed according 
                          to the various layers of the ANN model
        """        
        ...

        raise NotImplementedError
        
