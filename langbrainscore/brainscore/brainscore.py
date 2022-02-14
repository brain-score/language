
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
                 aggregate: typing.Union[None, typing.Callable] = np.mean) -> '_BrainScore':
        self.mapping = mapping
        self.metric = metric
        self.aggregate = aggregate or (lambda x: x)

    @staticmethod
    def _score(A, B, metric: Metric) -> np.ndarray:
        return metric(A, B)

    @lru_cache(maxsize=None)
    def score(self):
        result = self.mapping.fit()
        tests, preds = result['test'], result['pred']

        scores = []
        for A, B in zip(tests, preds):
            this_score = self._score(A, B, self.metric)
            this_score = xr.DataArray(this_score, dims=('neuroid',), coords={}).to_dataset(name='data')
    
            for k in A.to_dataset(name='data').drop_dims(['sampleid']).coords: #<- keeps only neuroid, and has no data
                this_score = this_score.assign_coords({k: ('neuroid', A[k].data)})
            scores.append(this_score)

        self.scores = xr.concat(scores, dim='sampleid')

        # self.scores = scores = np.array(scores, dtype='float64')
        logging.log(f'scores.shape: {self.scores.shape}, dims: {self.scores.dims}')

        # aggregating over splits
        return self.aggregate(np.nan_to_num(scores), axis=0)


    
