import typing
from abc import ABC, abstractmethod
from mimetypes import init

import numpy as np
import xarray as xr
from langbrainscore.interface.mapping import _Mapping
from langbrainscore.interface.metrics import _Metric


class _BrainScore(ABC):
    def __init__(self, mapping: _Mapping, metric: _Metric):
        pass

    @staticmethod
    def _score(A: xr.DataArray, B: xr.DataArray, metric: _Metric) -> np.ndarray:
        raise NotImplementedError

    def score(self) -> xr.DataArray:
        raise NotImplementedError
