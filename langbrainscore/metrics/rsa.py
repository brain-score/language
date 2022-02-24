import typing
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import pairwise_distances


class RSA:

    """
    This class compares the representational spaces
    of two embeddings of the same samples
    by calculating pairwise distances for each
    under a given similarity metric
    and then comparing those RSMs
    using a comparison metric
    """

    def __init__(
        self, distance_metric: str = "cosine", comparison_metric: str = "spearmanr"
    ) -> None:
        """

        valid similarity metrics: ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock',
        'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice',
        'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
        'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine']

        valid comparison metrics: ['pearsonr', 'spearmanr', 'kendalltau']

        """
        self._rdm = RDM(distance_metric)
        self._comparison_metric = comparison_metric

    def run(self, X: np.ndarray, Y: np.ndarray) -> typing.Tuple[float, float]:
        """
        accepts NxD two sample representations
        returns scalar comparison score and p-value
        """
        return tuple(
            globals()[self._comparison_metric](
                *(self._rdm.transform(rep) for rep in (X, Y))
            )
        )


class RDM:

    """
    This class utilizes a similarity metric
    to compute pairwise distances
    across samples over features
    within a representational space
    to produce an RDM.
    """

    def __init__(self, distance_metric: str) -> None:
        self._distance_metric = distance_metric

    def transform(self, sample_reps: np.ndarray) -> np.array:
        """
        accepts an N samples x D dimensions embedding
        returns the flattened N(N-1)/2 vector
        of the upper triangle of an N x N distance matrix
        of sample distances under distance metric
        """
        params = np.triu(
            pairwise_distances(sample_reps, metric=self._distance_metric), k=1
        ).flatten()
        return params[params != 0]
