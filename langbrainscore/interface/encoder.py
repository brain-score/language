from __future__ import annotations
from abc import ABC, abstractmethod
import typing
import numpy as np
import pandas as pd

import langbrainscore


class Encoder(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def encode(self, dataset: 'langbrainscore.dataset.DataSet') -> pd.DataFrame:
        return NotImplemented


class _BrainEncoder(Encoder):
    '''
    This class provides a wrapper around real-world brain data of various kinds
        which may be: 
            - neuroimaging [fMRI, PET]
            - physiological [ERP, MEG, ECOG]
            - behavioral [RT, Eye-tracking]
    across several subjects. The class implements `BrainEncoder.encode` which takes in
    a collection of stimuli (typically `np.array` or `list`) 
    '''

    # _dataset: langbrainscore.dataset.Dataset = None

    def __init__(self, dataset = None) -> None:
        # if not isinstance(dataset, langbrainscore.dataset.BrainDataset):
        #     raise TypeError(f"dataset must be of type `langbrainscore.dataset.BrainDataset`, not {type(dataset)}")
        # self._dataset = dataset
        NotImplemented

    # @property
    # def dataset(self) -> langbrainscore.dataset.Dataset:
    #     return self._dataset

    # @typing.overload
    # def encode(self, stimuli: typing.Union[np.array, list]): ...
    @abstractmethod
    def encode(self, dataset: 'langbrainscore.dataset.BrainDataSet' = None):
        """returns an "encoding" of stimuli (passed in as a BrainDataset)

        Args:
            stimuli (langbrainscore.dataset.BrainDataset):

        Returns:
            pd.DataFrame: neural recordings for each stimulus, multi-indexed 
                          by layer (trivially just 1 layer)
        """        
        
        NotImplemented


class _ANNEncoder(Encoder):
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
        
        NotImplemented
        
