
import typing
import numpy as np
import xarray as xr
from methodtools import lru_cache
from langbrainscore.interface.brainscore import _BrainScore
from langbrainscore.metrics import Metric
from langbrainscore.mapping import Mapping
from langbrainscore.utils import logging

class BrainScore(_BrainScore):

    def __init__(self, mapping: Mapping, metric: Metric, 
                 fold_aggregation: typing.Union[str, None] = 'mean',
                 run = True) -> '_BrainScore':
        self.mapping = mapping
        self.metric = metric 
        self.fold_aggregation = fold_aggregation 
        self.aggregate_methods_map = {
            None: self._no_aggregate,
            'mean': self._aggregate_mean,
        }

        if run:
            self.score()
            self.aggregate_scores()

    def __str__(self) -> str:
        return f'{self.scores.mean()}'

    def to_dataarray(self, aggregated=True):
        # returns the aggregated scores as an xarray
        return self.scores if aggregated else self.scores_across_folds
    
    def to_disk(self, aggregated=True):
        # outputs the aggregated (or not) object to disk
        # as a dataarray
        pass

    def aggregate_scores(self):
        """aggregates scores obtianed over 

        Args:
            dim (_type_): _description_
        """
        fn = self.aggregate_methods_map[self.fold_aggregation]
        self.scores = fn()

    def _no_aggregate(self):
        return self.scores_across_folds

    def _aggregate_mean(self):
        return self.scores_across_folds.mean(dim='cvfoldid')

    @staticmethod
    def _score(A, B, metric: Metric) -> np.ndarray:
        return metric(A, B)

    @lru_cache(maxsize=None)
    def score(self):
        result = self.mapping.fit()
        tests, preds = result['test'], result['pred']

        sample_scores = []
        # A, B are lists of xr DataArrays over timeids
        for A, B in zip(tests, preds):
            fold_scores = []
            for timeid_ix, timeid in enumerate(A.timeid.values):
                # A_time, B_time are xr DataArrays at a specific timeid
                A_time = A.isel(timeid=timeid)
                B_time = B[timeid_ix]
                this_score = self._score(A_time, B_time, self.metric)
                this_score = xr.DataArray(this_score, dims=('neuroid',), coords={}).to_dataset(name='data')
        
                # TODO
                # we want to package the score with the original metadata. we're trying below.
                for k in A.to_dataset(name='data').drop_dims(['sampleid', 'timeid']).coords: 
                    # ^- keeps only neuroid, and has no .data
                    this_score = this_score.assign_coords({k: ('neuroid', A[k].data)})
                
                fold_scores.append(this_score)

            sample_scores.append(xr.concat(fold_scores, dim='timeid'))
 

        scores = xr.concat(sample_scores, dim='cvfoldid').data
        scores = scores.assign_coords({'cvfoldid': ('cvfoldid', scores.cvfoldid.data)})

        for k in A.to_dataset(name='data').drop_dims(['sampleid', 'neuroid']).coords: 
            # ^- keeps only timeid, and has no .data
            scores = scores.assign_coords({k: ('timeid', A[k].data)})

        self.scores_across_folds = scores.fillna(0)

    
