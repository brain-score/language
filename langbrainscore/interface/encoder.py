from abc import ABC, abstractmethod

import langbrainscore
import xarray as xr


class Encoder(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def encode(self, dataset: 'langbrainscore.dataset.Dataset') -> xr.DataArray:
        raise NotImplementedError


class _BrainEncoder(Encoder):
    @abstractmethod
    def encode(self, dataset: 'langbrainscore.dataset.Dataset') -> xr.DataArray:
        """returns an "encoding" of stimuli (passed in as a Dataset)

        Args:
            langbrainscore.dataset.Dataset: brain dataset object

        Returns:
            xr.DataArray: contents of brain dataset
        """
        raise NotImplementedError


class _ANNEncoder(Encoder):
    @abstractmethod
    def encode(self, dataset: 'langbrainscore.dataset.Dataset') -> xr.DataArray:
        """returns embeddings of stimuli (passed in as a Dataset)

        Args:
            langbrainscore.dataset.Dataset: brain dataset object

        Returns:
            xr.DataArray: ANN representations of each stimulus in brain dataset

        """
        raise NotImplementedError
