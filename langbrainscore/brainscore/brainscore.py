import typing

import numpy as np
import xarray as xr
from tqdm.auto import tqdm

# from methodtools import lru_cache
from pathlib import Path

from langbrainscore.interface import (
    _BrainScore,
    _Mapping,
    _Metric,
    EncoderRepresentations,
)

# from langbrainscore.metrics import Metric
from langbrainscore.utils import logging
from langbrainscore.utils.xarray import collapse_multidim_coord, copy_metadata


class BrainScore(_BrainScore):
    scores = None
    ceilings = None
    nulls = []

    def __init__(
        self,
        X: typing.Union[xr.DataArray, EncoderRepresentations],
        Y: typing.Union[xr.DataArray, EncoderRepresentations],
        mapping: _Mapping,
        metric: _Metric,
        sample_split_coord: str = None,
        neuroid_split_coord: str = None,
        run=False,
    ) -> "BrainScore":
        """Initializes the [lang]BrainScore object using two encoded representations and a mapping
           class, and a metric for evaluation

        Args:
            X (typing.Union[xr.DataArray, EncoderRepresentations]): Either an xarray DataArray
                instance, or a wrapper object with a `.representations` attribute that stores the xarray
                DataArray
            Y (typing.Union[xr.DataArray, EncoderRepresentations]): see `X`
            mapping (_Mapping): _description_
            metric (_Metric): _description_
            run (bool, optional): _description_. Defaults to False.

        Returns:
            BrainScore: _description_
        """
        self.X = X.representations if hasattr(X, "representations") else X
        self.Y = Y.representations if hasattr(Y, "representations") else Y
        assert self.X.sampleid.size == self.Y.sampleid.size
        self.mapping = mapping
        self.metric = metric
        self._sample_split_coord = sample_split_coord
        self._neuroid_split_coord = neuroid_split_coord

        if run:
            self.run()

    def __str__(self) -> str:
        try:
            return f"{self.scores.mean()}"
        except AttributeError as e:
            raise ValueError(
                "missing scores. did you make a call to `score()` or `run()` yet?"
            )

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
    def _score(A, B, metric: _Metric) -> np.ndarray:
        return metric(A, B)

    # @lru_cache(maxsize=None)
    def score(
        self,
        ceiling=False,
        null=False,
        seed=0,
    ):
        """
        Computes The BrainScore™ (/s) using predictions/outputs returned by a
        Mapping instance which is a member attribute of a BrainScore instance
        """
        assert not (ceiling and null)
        sample_split_coord = self._sample_split_coord
        neuroid_split_coord = self._neuroid_split_coord

        if sample_split_coord:
            assert sample_split_coord in self.Y.coords

        if neuroid_split_coord:
            assert neuroid_split_coord in self.Y.coords

        X = self.X
        if null:
            y_shuffle = self.Y.copy()
            y_shuffle.data = np.random.default_rng(seed=seed).permutation(
                y_shuffle.data, axis=0
            )
            Y = y_shuffle
        else:
            Y = self.Y
        y_pred, y_true = self.mapping.fit_transform(X, Y, ceiling=ceiling)

        if not (ceiling or null):
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
                                        neuroid=(
                                            y_true_time_group[neuroid_split_coord]
                                            == neuroid
                                        )
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

        if not (ceiling or null):
            self.scores = scores
        elif ceiling:
            self.ceilings = scores
        else:
            self.nulls.append(
                scores.expand_dims(dim={"iter": [seed]}, axis=-1).assign_coords(
                    iter=[seed]
                )
            )

    def ceiling(self):  # , sample_split_coord=None, neuroid_split_coord=None):
        logging.log("Calculating ceiling.", type="INFO")
        self.score(
            ceiling=True,
            # sample_split_coord=self._sample_split_coord,
            # neuroid_split_coord=neuroid_split_coord,
        )

    def null(
        self,
        # sample_split_coord=None, neuroid_split_coord=None,
        iters=100,
    ):
        for i in tqdm([*range(iters)], desc="Running null permutations"):
            self.score(
                null=True,
                # sample_split_coord=sample_split_coord,
                # neuroid_split_coord=neuroid_split_coord,
                seed=i,
            )
        self.nulls = xr.concat(self.nulls, dim="iter")

    def run(
        self,
        sample_split_coord=None,
        neuroid_split_coord=None,
        calc_nulls=False,
        iters=100,
    ):
        self.score(
            sample_split_coord=sample_split_coord,
            neuroid_split_coord=neuroid_split_coord,
        )
        self.ceiling(
            sample_split_coord=sample_split_coord,
            neuroid_split_coord=neuroid_split_coord,
        )
        if calc_nulls:
            self.null(
                sample_split_coord=sample_split_coord,
                neuroid_split_coord=neuroid_split_coord,
                iters=iters,
            )
            return {
                "scores": self.scores,
                "ceilings": self.ceilings,
                "nulls": self.nulls,
            }
        return {"scores": self.scores, "ceilings": self.ceilings}
