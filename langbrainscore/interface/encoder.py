from abc import ABC, abstractmethod

import xarray as xr
from langbrainscore.dataset import Dataset


class _Encoder(ABC):
    """
    Interface for *Encoder classes.
    Must implement an `encode` method that operates on a Dataset object.
    """

    @staticmethod
    def _check_dataset_interface(dataset):
        """
        confirms that dataset adheres to `langbrainscore.dataset.Dataset` interface.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"dataset must be of type `langbrainscore.dataset.Dataset`, not {type(dataset)}"
            )

    @abstractmethod
    def encode(self, dataset: Dataset) -> xr.DataArray:
        raise NotImplementedError


class _ModelEncoder(_Encoder):
    def __init__(self, model_id: str) -> "_ModelEncoder":
        self._model_id = model_id

    @abstractmethod
    def encode(self, dataset: Dataset) -> xr.DataArray:
        """
        returns embeddings of stimuli (passed in as a Dataset)

        Args:
            langbrainscore.dataset.Dataset: brain dataset object

        Returns:
            xr.DataArray: Model representations of each stimulus in brain dataset
        """
        raise NotImplementedError
