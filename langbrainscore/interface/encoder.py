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


    def __repr__(self) -> str:
        return f"<{self.__class__} {self._model_id}>"
    def __str__(self) -> str:
        return repr(self)


    @abstractmethod
    def encode(self, dataset: Dataset) -> xr.DataArray:
        """
        returns computed representations for stimuli passed in as a `Dataset` object

        Args:
            langbrainscore.dataset.Dataset: a Dataset object with a member `xarray.DataArray` 
                instance containing stimuli

        Returns:
            xr.DataArray: Model representations of each stimulus in brain dataset
        """
        raise NotImplementedError
