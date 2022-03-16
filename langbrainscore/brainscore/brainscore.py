import typing

import numpy as np
import xarray as xr
from langbrainscore.interface.brainscore import _BrainScore
from langbrainscore.mapping import Mapping
from langbrainscore.metrics import Metric
from langbrainscore.utils import logging
from methodtools import lru_cache


class BrainScore(_BrainScore):
    def __init__(
        self,
        mapping: Mapping,
        metric: Metric,
        fold_aggregation: typing.Union[str, None] = "mean",
        run=True,
    ) -> "BrainScore":
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
        scores_per_neuroid = []
        # returns generator over neuroids of (tests, preds, alphas) tuples
        # tests, preds are xarrays, and alphas is a list of dicts per cvfold.
        # the dict is indexed using timeids

        # NOTE: need to modify this to support IdentityMap scenario
        for result in self.mapping.fit():

            tests, preds, alphas = result["test"], result["pred"], result["alphas"]
            scores_per_fold = []
            # A, B are lists of xr DataArrays over timeids
            for cvid, (A, B, T) in enumerate(zip(tests, preds, alphas)):
                scores_per_timeid = []  # this is a list of n_folds xarrays
                for timeid in A.timeid.values:
                    # A_time, B_time are xr DataArrays at a specific timeid
                    A_time = A.sel(timeid=0)  # TODO RNNs ;_;
                    B_time = B.sel(timeid=timeid)  # B[timeid_ix]
                    alpha_time = T[timeid]
                    timeid_score_scalar = self._score(A_time, B_time, self.metric)
                    timeid_score = xr.DataArray(
                        timeid_score_scalar,
                        dims=("neuroid", "timeid"),
                        coords={
                            "neuroid": ("neuroid", B_time.neuroid.values.reshape(-1)),
                            # ^ we want to extract [int] from 0-d array (scalar array)
                            "timeid": ("timeid", [timeid]),
                            # WIP: alpha
                            "alpha": (
                                ("timeid", "neuroid"),
                                np.array(alpha_time).reshape(1, 1),
                            ),
                        },
                    )
                    scores_per_timeid.append(timeid_score)

                fold_score = (
                    xr.concat(scores_per_timeid, dim="timeid")
                    .expand_dims("cvfoldid", 0)
                    .assign_coords(cvfoldid=("cvfoldid", [cvid]))
                )
                scores_per_fold.append(fold_score)

            neuroid_score = xr.concat(scores_per_fold, dim="cvfoldid")
            scores_per_neuroid.append(neuroid_score)

        self.scores_unfolded = xr.concat(scores_per_neuroid, dim="neuroid")

        # fill back in metadata coords for neuroid and timeid from Y
        for k in (
            self.mapping.Y.to_dataset(name="data")
            .drop_dims(["sampleid", "timeid"])
            .coords
        ):
            self.scores_unfolded = self.scores_unfolded.assign_coords(
                {k: ("neuroid", self.mapping.Y[k].data)}
            )
        for k in (
            self.mapping.Y.to_dataset(name="data")
            .drop_dims(["sampleid", "neuroid"])
            .coords
        ):
            self.scores_unfolded = self.scores_unfolded.assign_coords(
                {k: ("timeid", self.mapping.Y[k].data)}
            )

        # aggregate scores across cv folds to obtain scores per neuroid and per timeid
        self.aggregate_scores()

        return self.scores
