
from abc import ABC, abstractmethod
from mimetypes import init
import typing
import numpy as np

from langbrainscore.interface.mapping import _Mapping
from langbrainscore.interface.metrics import _Metric

class _BrainScore(ABC):
    
    mapping: _Mapping = None
    ''' an instance of the Mapping class with a particular 
        split, stratification, and data'''

    def __init__(self, mapping: _Mapping, metric: _Metric) -> '_BrainScore':
        NotImplemented

    @staticmethod
    def _score(A, B, metric: typing.Union[str, typing.Any]) -> np.ndarray:
        NotImplemented

    def score(self):
        '''
        Scores the Mapping instance we have according to a metric this object was
        instantiated with
        '''
        NotImplemented




