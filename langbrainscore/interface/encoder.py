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
        ...

    # @typing.overload
    # def encode(self, stimuli: typing.Union[np.array, list]): ...
    def encode(self, dataset: 'langbrainscore.dataset.BrainDataset'):
        """returns an "encoding" of stimuli (passed in as a BrainDataset)

        Args:
            stimuli (langbrainscore.dataset.BrainDataset):

        Returns:
            pd.DataFrame: neural recordings for each stimulus, multi-indexed 
                          by layer (trivially just 1 layer)
        """        
        df = pd.DataFrame(dataset.recorded_data)
        df.columns = pd.MultiIndex.from_tuples([(neuroid_id, 0) for neuroid_id in range(dataset.num_neuroids)])
        return df.to_numpy()


class ANNEncoder(Encoder):
    def __init__(self) -> None:
        super().__init__(self)
        pass


    def encode(self, dataset: 'langbrainscore.dataset.DataSet'):
        """[summary]

        Args:
            stimuli (langbrainscore.dataset.DataSet): [description]

        Returns:
            pd.DataFrame: neural recordings for each stimulus, multi-indexed according 
                          to the various layers of the ANN model
        """        
        ...


        # df = pd.DataFrame(dataset)
        # df.columns = pd.MultiIndex.from_tuples([(neuroid_id, 0) for neuroid_id in range(dataset.num_neuroids)])
