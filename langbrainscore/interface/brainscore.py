import typing
from abc import ABC, abstractmethod
from mimetypes import init

import numpy as np
import xarray as xr
from langbrainscore.mapping import Mapping
from langbrainscore.metrics import Metric


class _BrainScore(ABC):
    def __init__(
        self, X: xr.DataArray, Y: xr.DataArray, mapping: Mapping, metric: Metric
    ):
        pass

    def score(self) -> xr.DataArray:
        """
        applies mapping to (X, Y), then evaluates using metric

        Returns:
            xr.DataArray: scores
        """
        raise NotImplementedError
