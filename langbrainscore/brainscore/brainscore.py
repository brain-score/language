import typing

import numpy as np
import xarray as xr
from langbrainscore.interface import _BrainScore, _Mapping
from langbrainscore.metrics import Metric
from langbrainscore.utils import logging
from langbrainscore.utils.xarray import copy_metadata
from methodtools import lru_cache
from pathlib import Path

class BrainScore(_BrainScore):
    def __init__(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        mapping: _Mapping,
        metric: Metric,
        run=True,
    ) -> "BrainScore":
        assert X.sampleid.size == Y.sampleid.size
        self.X = X
        self.Y = Y
        self.mapping = mapping
        self.metric = metric
        if run:
            self.score()

    def __repr__(self) -> str:
        return f'<{self.__class__} ({self.mapping}, {self.metric}, {str(self)})>'

    def __str__(self) -> str:
        return f"{self.scores.mean()}"

    
    def to_netcdf(self, filename):
        '''
        outputs the xarray.DataArray object for 'scores' to a netCDF file
        identified by `filename`. if it already exists, overwrites it.
        '''
        if Path(filename).expanduser().resolve().exists():
            logging.log(f'{filename} already exists. overwriting.', type='WARN')
        self.scores.to_netcdf(filename)


    def load_netcdf(self, filename):
        '''
        loads a netCDF object that contains an xarray instance for 'scores' from
        a file at `filename`.
        '''
        self.scores = xr.load_dataarray(filename)


    # TODO: marked for removal
    # def to_dataset(self) -> xr.Dataset:
    #     return xr.Dataset({"Y": self.Y, "Y_pred": self.Y_pred, "scores": self.scores})

    # def to_disk(self):
    #     X = self.X
    #     dataset = self.to_dataset()
    #     pass


    @staticmethod
    def _score(A, B, metric: Metric) -> np.ndarray:
        return metric(A, B)

    @lru_cache(maxsize=None)
    def score(self):
        """
        Computes The BrainScore™ (/s) using predictions/outputs returned by a
        Mapping instance which is a member attribute of a BrainScore instance
        """

        y_pred, y_true = self.mapping.fit_transform(self.X, self.Y)

        self.Y_pred = y_pred
        if y_pred.shape == y_true.shape:  # not IdentityMap
            self.Y_pred = copy_metadata(self.Y_pred, self.Y, "sampleid")
            self.Y_pred = copy_metadata(self.Y_pred, self.Y, "neuroid")
            self.Y_pred = copy_metadata(self.Y_pred, self.Y, "timeid")

        scores_over_time = []
        for timeid in y_true.timeid.values:

            y_pred_time = y_pred.sel(timeid=timeid).transpose("sampleid", "neuroid")
            y_true_time = y_true.sel(timeid=timeid).transpose("sampleid", "neuroid")

            score_per_timeid = self._score(y_pred_time, y_true_time, self.metric)

            if len(score_per_timeid) == 1:  # e.g., RSA, CKA
                neuroids = [np.nan]
            else:
                neuroids = y_true_time.neuroid.data

            scores_over_time.append(
                xr.DataArray(
                    score_per_timeid.reshape(-1, 1),
                    dims=("neuroid", "timeid"),
                    coords={
                        "neuroid": ("neuroid", neuroids),
                        "timeid": ("timeid", [timeid]),
                    },
                )
            )

        scores = xr.concat(scores_over_time, dim="timeid")
        
        if scores.neuroid.size > 1:  # not RSA
            scores = copy_metadata(scores, self.Y, "neuroid")
        scores = copy_metadata(scores, self.Y, "timeid")

        self.scores = scores
        return self.scores
