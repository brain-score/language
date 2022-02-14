from langbrainscore.interface.encoder import _BrainEncoder
import numpy as np

from langbrainscore.utils import logging


class BrainEncoder(_BrainEncoder):
    '''
    This class provides a wrapper around real-world brain data of various kinds
        which may be: 
            - neuroimaging [fMRI, PET]
            - physiological [ERP, MEG, ECOG]
            - behavioral [RT, Eye-tracking]
    across several subjects. The class implements `BrainEncoder.encode` which takes in
    a collection of stimuli (typically `np.array` or `list`) 
    '''

    _dataset: 'langbrainscore.dataset.Dataset' = None

    def __init__(self, dataset = None) -> None:
        # if not isinstance(dataset, langbrainscore.dataset.BrainDataset):
        #     raise TypeError(f"dataset must be of type `langbrainscore.dataset.BrainDataset`, not {type(dataset)}")
        self._dataset = dataset

    @property
    def dataset(self) -> 'langbrainscore.dataset.Dataset':
        return self._dataset

    # @typing.overload
    # def encode(self, stimuli: typing.Union[np.array, list]): ...
    def encode(self, dataset: 'langbrainscore.dataset.BrainDataSet' = None,
               average_time = True):
        """returns an "encoding" of stimuli (passed in as a BrainDataset)

        Args:
            stimuli (langbrainscore.dataset.BrainDataset):

        Returns:
            pd.DataFrame: neural recordings for each stimulus, multi-indexed 
                          by layer (trivially just 1 layer)
        """        
        
        dataset = dataset or self.dataset

        if (timeid_dims := dataset._dataset.dims['timeid']) >= 1:
            # TODO: revisit this
            if average_time:
                return dataset._dataset.mean('timeid')
            return dataset._dataset
        # elif timeid_dims == 1:
        #     return dataset._dataset.squeeze('timeid')
        else:
            raise ValueError(f'timeid has invalid shape {timeid_dims}')