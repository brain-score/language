import typing

import numpy as np
import xarray as xr
from methodtools import lru_cache

from langbrainscore.interface import _BrainScore, _Mapping
from langbrainscore.metrics import Metric
from langbrainscore.utils import logging
from langbrainscore.utils.xarray import collapse_multidim_coord, copy_metadata


class BrainScore(_BrainScore):
    def __init__(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        mapping: _Mapping,
        metric: Metric,
        run=False,
    ) -> "BrainScore":
        assert X.sampleid.size == Y.sampleid.size
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
    def score(self, score_split_coord=None):
        """
        Computes The BrainScore™ (/s) using predictions/outputs returned by a
        Mapping instance which is a member attribute of a BrainScore instance
        """

        if score_split_coord:
            assert score_split_coord in self.Y.coords

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

            if score_split_coord:
                if score_split_coord not in y_true_time.sampleid.coords:
                    y_pred_time = collapse_multidim_coord(
                        y_pred_time, score_split_coord, "sampleid"
                    )
                    y_true_time = collapse_multidim_coord(
                        y_true_time, score_split_coord, "sampleid"
                    )
                score_splits = y_pred_time.sampleid.groupby(score_split_coord).groups
            else:
                score_splits = [0]

            scores_over_time_group = []
            for scoreid in score_splits:

                if score_split_coord:
                    y_pred_time_group = y_pred_time.isel(
                        sampleid=y_pred_time[score_split_coord] == scoreid
                    )
                    y_true_time_group = y_true_time.isel(
                        sampleid=y_true_time[score_split_coord] == scoreid
                    )
                else:
                    y_pred_time_group = y_pred_time
                    y_true_time_group = y_true_time

                score_per_time_group = self._score(
                    y_pred_time_group, y_true_time_group, self.metric
                )

                if len(score_per_time_group) == 1:  # e.g., RSA, CKA
                    neuroids = [np.nan]
                else:
                    neuroids = y_true_time_group.neuroid.data

                scores_over_time_group.append(
                    xr.DataArray(
                        score_per_time_group.reshape(1, -1, 1),
                        dims=("scoreid", "neuroid", "timeid"),
                        coords={
                            "scoreid": ("scoreid", [scoreid]),
                            "neuroid": ("neuroid", neuroids),
                            "timeid": ("timeid", [timeid]),
                        },
                    )
                )

            scores_over_time.append(xr.concat(scores_over_time_group, dim="scoreid"))

        scores = xr.concat(scores_over_time, dim="timeid")

        if scores.neuroid.size > 1:  # not RSA
            scores = copy_metadata(scores, self.Y, "neuroid")

        scores = copy_metadata(scores, self.Y, "timeid")

        self.scores = scores

        return self.scores
