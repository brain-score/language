import numpy as np
from brainscore_language.interface import _MatrixMetric, _VectorMetric
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import accuracy_score, mean_squared_error, pairwise_distances


class PearsonR(_VectorMetric):
    @staticmethod
    def _score(x: np.ndarray, y: np.ndarray) -> np.float:
        r, p = pearsonr(x, y)
        return r


class SpearmanRho(_VectorMetric):
    @staticmethod
    def _score(x: np.ndarray, y: np.ndarray) -> np.float:
        rho, p = spearmanr(x, y)
        return rho


class KendallTau(_VectorMetric):
    @staticmethod
    def _score(x: np.ndarray, y: np.ndarray) -> np.float:
        tau, p = kendalltau(x, y)
        return tau


class FisherCorr(_VectorMetric):
    @staticmethod
    def _score(x: np.ndarray, y: np.ndarray) -> np.float:
        r, p = pearsonr(x, y)
        corr = np.arctanh(r)
        return corr


class RMSE(_VectorMetric):
    @staticmethod
    def _score(x: np.ndarray, y: np.ndarray) -> np.float:
        loss = mean_squared_error(x, y, squared=False)
        return loss


class ClassificationAccuracy(_VectorMetric):
    @staticmethod
    def _score(x: np.ndarray, y: np.ndarray) -> np.float:
        score = accuracy_score(x, y, normalize=True)
        return score


class RSA(_MatrixMetric):
    """
    evaluates representational similarity between two matrices for a given
    distance measure and vector comparison metric
    """

    def __init__(self, distance="correlation", comparison=PearsonR()):
        """
        args:
            string: distance (anything accepted by sklearn.metrics.pairwise_distances)
            _VectorMetric: comparison
        """
        self._distance = distance
        self._comparison = comparison
        super().__init__()

    def _score(self, X: np.ndarray, Y: np.ndarray) -> np.float:
        X_rdm = pairwise_distances(X, metric=self._distance)
        Y_rdm = pairwise_distances(Y, metric=self._distance)
        if any([m.shape[1] == 1 for m in (X, Y)]):  # can't calc 1D corr dists
            X_rdm[np.isnan(X_rdm)] = 0
            Y_rdm[np.isnan(Y_rdm)] = 0
        indices = np.triu_indices(X_rdm.shape[0], k=1)
        score = self._comparison(X_rdm[indices], Y_rdm[indices])
        return score


# inspired by https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py
class CKA(_MatrixMetric):
    """
    evaluates centered kernel alignment distance between two matrices
    currently only implements linear kernel
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _center(K):
        N = K.shape[0]
        U = np.ones([N, N])
        I = np.eye(N)
        H = I - U / N
        centered = H @ K @ H
        return centered

    def _HSIC(self, A, B):
        L_A = A @ A.T
        L_B = B @ B.T
        HSIC = np.sum(self._center(L_A) * self._center(L_B))
        return HSIC

    def _score(self, X: np.ndarray, Y: np.ndarray) -> np.float:
        HSIC_XY = self._HSIC(X, Y)
        HSIC_XX = self._HSIC(X, X)
        HSIC_YY = self._HSIC(Y, Y)
        score = HSIC_XY / (np.sqrt(HSIC_XX) * np.sqrt(HSIC_YY))
        return score
