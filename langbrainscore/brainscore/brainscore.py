import typing

import numpy as np
import xarray as xr
from langbrainscore.interface.brainscore import _BrainScore
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
        fold_aggregation: typing.Union[str, None] = "mean",
        run=True,
    ) -> "BrainScore":
        self.X = X
        self.Y = Y
        self.mapping = mapping
        self.metric = metric
        self.fold_aggregation = fold_aggregation
        self.aggregate_methods_map = {
            None: self._no_aggregate,
            "mean": self._aggregate_mean,
        }

        if run:
            self.score()

    def __str__(self) -> str:
        return f"{self.scores.mean()}"

    def to_dataarray(self, aggregated: bool = True) -> xr.DataArray:
        # returns the aggregated scores as an xarray
        return self.scores if aggregated else self.scores_across_folds

    def to_disk(self, aggregated=True):
        # outputs the aggregated (or not) object to disk
        # as a dataarray
        pass

    def aggregate_scores(self):
        """
        aggregates scores obtianed over

        Args:
            dim (_type_): _description_
        """
        fn = self.aggregate_methods_map[self.fold_aggregation]
        self.scores = fn()

    def _no_aggregate(self):
        return self.scores_unfolded

    def _aggregate_mean(self):
        return self.scores_unfolded.mean(dim="cvfoldid")

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
