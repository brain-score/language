import typing
from abc import ABC, abstractmethod

import numpy as np


class _Metric(ABC):
    """
    checks that two arrays are comparable for a given similarity metric,
    then applies that metric to those inputs and returns score(s)

    Args:
        np.array: X
        np.array: Y

    Returns:
        np.array: score(s)

    Raises:
        ValueError: X and Y must be 1D or 2D arrays.
        ValueError: X and Y must have the same number of samples.
        ValueError: for most metrics, X and Y must have same number of dimensions.

    """

    def __init__(self):
        pass

    def __call__(
        self,
        X: typing.Union[np.array, np.ndarray],
        Y: typing.Union[np.array, np.ndarray],
    ) -> typing.Union[np.float, np.array]:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if any(y.ndim != 2 for y in [X, Y]):
            raise ValueError("X and Y must be 1D or 2D arrays.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples.")
        if self.__class__.__name__ not in ["RSA", "CKA"]:
            if X.shape[1] != Y.shape[1]:
                raise ValueError("X and Y must have the same number of dimensions.")
        return self._apply_metric(X, Y)

    @abstractmethod
    def _apply_metric(
        self, X: np.ndarray, Y: np.ndarray,
    ) -> typing.Union[np.float, np.array]:
        raise NotImplementedError


class _VectorMetric(_Metric):
    """
    subclass of _Metric that applies relevant vector similarity metric
    along each column of the input arrays.
    """

    def __init__(self, reduction=np.mean):
        """
        args:
            callable: reduction (can also be None or False)

        raises:
            TypeError: if reduction argument is not callable.
        """
        if reduction:
            if not callable(reduction):
                raise TypeError("Reduction argument must be callable.")
        self._reduction = reduction
        super().__init__()

    def _apply_metric(
        self, X: np.ndarray, Y: np.ndarray,
    ) -> typing.Union[np.float, np.array]:
        """
        internal function that applies scoring function along each array dimension
        and then optionally applies a reduction, e.g., np.mean

        args:
            np.ndarray: X
            np.ndarray: Y

        """
        scores = np.zeros(X.shape[1])
        for i in range(scores.size):
            scores[i] = self._score(X[:, i], Y[:, i])
        if self._reduction:
            return self._reduction(scores)
        return scores

    @abstractmethod
    def _score(self, X: np.array, Y: np.array) -> np.float:
        raise NotImplementedError


class _MatrixMetric(_Metric):
    """
    interface for similarity metrics that operate over entire matrices, e.g., RSA
    """

    def __init__(self):
        super().__init__()

    def _apply_metric(self, X: np.ndarray, Y: np.ndarray) -> np.float:
        score = self._score(X, Y)
        return score

    @abstractmethod
    def _score(self, X: np.ndarray, Y: np.ndarray) -> np.float:
        raise NotImplementedError