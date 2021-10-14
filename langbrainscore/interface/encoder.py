from abc import ABC, abstractmethod
import typing
import numpy as np

import langbrainscore

class Encoder(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def encode(self, X: typing.Union[np.array, list]) -> np.array:
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

    def __init__(self) -> None:
        super().__init__(self)
        pass

    @typing.overload
    def encode(self, stimuli: typing.Union[np.array, list]):
    def encode(self, X: typing.Union[np.array, list]): ...
        return stimuli


class SilicoEncoder(Encoder):
    def __init__(self) -> None:
        super().__init__(self)
        pass
