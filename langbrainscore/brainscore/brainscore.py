import typing

import numpy as np
import xarray as xr
# from methodtools import lru_cache
from pathlib import Path

from langbrainscore.interface import _BrainScore, _Mapping, _Metric
from langbrainscore.metrics import Metric
from langbrainscore.utils import logging
from langbrainscore.utils.xarray import collapse_multidim_coord, copy_metadata


class BrainScore(_BrainScore):
    scores = None
    ceilings = None

    def __init__(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        mapping: _Mapping,
        metric: _Metric,
        run=False,
    ) -> "BrainScore":
        assert X.sampleid.size == Y.sampleid.size
        self.X = X
        self.Y = Y
        self.mapping = mapping
        self.metric = metric
        if run:
            self.run()


    def __str__(self) -> str:
        return f"{self.scores.mean()}"

    def to_netcdf(self, filename):
        """
        outputs the xarray.DataArray object for 'scores' to a netCDF file
        identified by `filename`. if it already exists, overwrites it.
        """
        if Path(filename).expanduser().resolve().exists():
            logging.log(f"{filename} already exists. overwriting.", type="WARN")
        self.scores.to_netcdf(filename)

    def load_netcdf(self, filename):
        """
        loads a netCDF object that contains an xarray instance for 'scores' from
        a file at `filename`.
        """
        self.scores = xr.load_dataarray(filename)


    @staticmethod
    def _score(A, B, metric: Metric) -> np.ndarray:
        return metric(A, B)

    # @lru_cache(maxsize=None)
    def score(self, ceiling=False, sample_split_coord=None, neuroid_split_coord=None):
        """
        Computes The BrainScore™ (/s) using predictions/outputs returned by a
        Mapping instance which is a member attribute of a BrainScore instance
        """
        if not ceiling:
            if self.scores is not None:
                return self.scores
        else:
            if self.ceilings is not None:
                return self.ceilings

        if sample_split_coord:
            assert sample_split_coord in self.Y.coords

        if neuroid_split_coord:
            assert neuroid_split_coord in self.Y.coords

        y_pred, y_true = self.mapping.fit_transform(self.X, self.Y, ceiling=ceiling)

        if not ceiling:
            self.Y_pred = y_pred
            if y_pred.shape == y_true.shape:  # not IdentityMap
                self.Y_pred = copy_metadata(self.Y_pred, self.Y, "sampleid")
                self.Y_pred = copy_metadata(self.Y_pred, self.Y, "neuroid")
                self.Y_pred = copy_metadata(self.Y_pred, self.Y, "timeid")

        scores_over_time = []
        for timeid in y_true.timeid.values:

            y_pred_time = y_pred.sel(timeid=timeid).transpose("sampleid", "neuroid")
            y_true_time = y_true.sel(timeid=timeid).transpose("sampleid", "neuroid")

            if sample_split_coord:
                if sample_split_coord not in y_true_time.sampleid.coords:
                    y_pred_time = collapse_multidim_coord(
                        y_pred_time, sample_split_coord, "sampleid"
                    )
                    y_true_time = collapse_multidim_coord(
                        y_true_time, sample_split_coord, "sampleid"
                    )
                score_splits = y_pred_time.sampleid.groupby(sample_split_coord).groups
            else:
                score_splits = [0]

            scores_over_time_group = []
            for scoreid in score_splits:

                if sample_split_coord:
                    y_pred_time_group = y_pred_time.isel(
                        sampleid=y_pred_time[sample_split_coord] == scoreid
                    )
                    y_true_time_group = y_true_time.isel(
                        sampleid=y_true_time[sample_split_coord] == scoreid
                    )
                else:
                    y_pred_time_group = y_pred_time
                    y_true_time_group = y_true_time

                neuroids = []
                if y_pred.shape != y_true.shape and neuroid_split_coord:  # IdentityMap
                    if neuroid_split_coord:
                        if neuroid_split_coord not in y_true_time_group.neuroid.coords:
                            y_true_time_group = collapse_multidim_coord(
                                y_true_time_group, neuroid_split_coord, "neuroid"
                            )
                        neuroid_splits = y_true_time_group.neuroid.groupby(
                            neuroid_split_coord
                        ).groups
                        score_per_time_group = []
                        for neuroid in neuroid_splits:
                            score_per_time_group.append(
                                self._score(
                                    y_pred_time_group,
                                    y_true_time_group.isel(
                                        neuroid=y_true_time_group[neuroid_split_coord]
                                        == neuroid
                                    ),
                                    self.metric,
                                )
                            )
                            neuroids.append(neuroid)
                        score_per_time_group = np.array(score_per_time_group)
                else:
                    score_per_time_group = self._score(
                        y_pred_time_group, y_true_time_group, self.metric
                    )

                if neuroids:
                    pass
                elif len(score_per_time_group) == 1:  # e.g., RSA, CKA, w/o split
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

        if scores.neuroid.size == self.Y.neuroid.size:  # not RSA, CKA, etc.
            scores = copy_metadata(scores, self.Y, "neuroid")
        scores = copy_metadata(scores, self.Y, "timeid")

        if not ceiling:
            self.scores = scores
        else:
            self.ceilings = scores


    def ceiling(self, sample_split_coord=None, neuroid_split_coord=None):
        return self.score(
            ceiling=True,
            sample_split_coord=sample_split_coord,
            neuroid_split_coord=neuroid_split_coord,
        )

    def run(self, sample_split_coord=None, neuroid_split_coord=None):
        scores = self.score(
            sample_split_coord=sample_split_coord,
            neuroid_split_coord=neuroid_split_coord,
        )
        ceilings = self.ceiling(
            sample_split_coord=sample_split_coord,
            neuroid_split_coord=neuroid_split_coord,
        )
        return {"scores": scores, "ceilings": ceilings}
