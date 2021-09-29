from abc import ABC, abstractmethod
from typing import Tuple

import xarray as xr
from langbrainscore.interface.cacheable import _Cacheable


class _Mapping(_Cacheable, ABC):
    # class _Mapping(ABC):
    """
    object that defines and applies map between two xarrays with the same number of samples
    """

    def __init__(self):
        pass

    def fit_transform(
        self, X: xr.DataArray, Y: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        takes in two xarrays with a shared set of samples and returns a new
        pair of xarrays (Y_pred, Y_true) to be compared with a metric.T

        Y_pred is either derived from a learned mapping on X or can be X itself
        when the downstream metric supports comparison of matrices with
        different dimensions, e.g, RSA, CKA

        args:
            xr.DataArray: X
            xr.DataArray: Y

        returns:
            xr.DataArray: Y_pred
            xr.DataArray: Y_true
        """
        raise NotImplementedError
