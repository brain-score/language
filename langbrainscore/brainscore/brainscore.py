import typing

import numpy as np
import xarray as xr
from langbrainscore.interface import _BrainScore
from langbrainscore.mapping import Mapping
from langbrainscore.metrics import Metric
from langbrainscore.utils import logging
from langbrainscore.utils.xarray import copy_metadata
from methodtools import lru_cache


class BrainScore(_BrainScore):
    def __init__(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        mapping: Mapping,
        metric: Metric,
        run=True,
    ) -> "BrainScore":
        self.X = X
        self.Y = Y
        self.mapping = mapping
        self.metric = metric
        if run:
            self.score()

    def __str__(self) -> str:
        return f"{self.scores.mean()}"

    def to_dataset(self) -> xr.Dataset:
        return xr.Dataset({"Y": self.Y, "Y_pred": self.Y_pred, "scores": self.scores})

    def to_disk(self):
        X = self.X
        dataset = self.to_dataset()
        # TODO
        pass

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

        if y_pred.shape == y_true.shape:  # not IdentityMap
            y_pred = copy_metadata(y_pred, self.Y, "sampleid")
            y_pred = copy_metadata(y_pred, self.Y, "neuroid")
            y_pred = copy_metadata(y_pred, self.Y, "timeid")

        self.Y_pred = y_pred

        scores_over_time = []
        for timeid in y_true.timeid.values:

            y_pred_time = y_pred.sel(timeid=timeid).transpose("sampleid", "neuroid")
            y_true_time = y_true.sel(timeid=timeid).transpose("sampleid", "neuroid")

            score_per_timeid = self._score(y_pred_time, y_true_time, self.metric)

            if len(score_per_timeid) < y_true_time.shape[1]:  # e.g., RSA
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
