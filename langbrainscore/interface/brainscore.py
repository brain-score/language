import typing
from abc import ABC, abstractmethod
from mimetypes import init

import numpy as np
import xarray as xr
from langbrainscore.interface.mapping import _Mapping
from langbrainscore.interface.metric import _Metric
from langbrainscore.interface.cacheable import _Cacheable


class _BrainScore(_Cacheable, ABC):
    # class _BrainScore(ABC):
    """
    evaluates a `Mapping` of `X` and `Y` using `Metric`
    """

    def __init__(
        self, X: xr.DataArray, Y: xr.DataArray, mapping: _Mapping, metric: _Metric
    ):
        pass

    def score(self) -> xr.DataArray:
        """
        applies mapping to (X, Y), then evaluates using metric

        Returns:
            xr.DataArray: scores
        """
        raise NotImplementedError
