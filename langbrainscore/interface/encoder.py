'''
Interface and (partial) base implementation classes for Encoders and EncodedRepresentations
'''

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import xarray as xr
from langbrainscore.dataset import Dataset
from langbrainscore.interface.cacheable import _Cacheable


class _Encoder(_Cacheable, ABC):
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
    def __init__(self, model_id: str, **kwargs) -> "_ModelEncoder":
        """This class is intended to be an interface for all ANN subclasses,
            including HuggingFaceEncoder, and, in the future, other kinds of
            ANN encoders

        Args:
            model_id (str): _description_

        Returns:
            _ModelEncoder: _description_
        """

        self._model_id = model_id
        for k, v in kwargs.items():
            setattr(self, k, v)


    @abstractmethod
    def encode(self, dataset: Dataset) -> xr.DataArray:
        """
        returns computed representations for stimuli passed in as a `Dataset` object

        Args:
            langbrainscore.dataset.Dataset: a Dataset object with a member `xarray.DataArray` 
                instance (`Dataset._xr_obj`) containing stimuli

        Returns:
            xr.DataArray: Model representations of each stimulus in brain dataset
        """
        raise NotImplementedError


@dataclass(repr=False, eq=False, frozen=True)
class EncoderRepresentations(_Cacheable):
    '''
    a class to hold the encoded representations output from an `_Encoder.encode` method
    '''
    dataset: Dataset # pointer to the dataset these are the EncodedRepresentations of
    representations: xr.DataArray # the xarray holding representations

    context_dimension: str = None
    bidirectional: bool = False
    emb_case: typing.Union[str, None] = "lower"
    emb_aggregation: typing.Union[str, None, typing.Callable] = "last"
    emb_preproc: typing.Tuple[str] = ()

    def __getattr__(self, __name: str) -> typing.Any:
        '''falls back on the xarray object in case of a NameError using __getattribute__
            on this object'''
        try:
            return getattr(self.representations, __name)
        except AttributeError:
            raise AttributeError(f'no attribute called `{__name}`')
        