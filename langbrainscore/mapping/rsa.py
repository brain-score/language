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

    def __init__(self, similarity_metric: str, comparison_metric: str) -> None:
        """

        valid similarity metrics: ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock',
        'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice',
        'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
        'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine']

        valid comparison metrics: ['pearsonr', 'spearmanr', 'kendalltau']

        """
        self._rdm = RDM(similarity_metric)
        self._comparison_metric = comparison_metric

    def run(self, X: np.ndarray, Y: np.ndarray) -> float:
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

    def __init__(self, similarity_metric: str) -> None:
        self._similarity_metric = similarity_metric

    def transform(self, sample_reps: np.ndarray) -> np.ndarray:
        """
        accepts an N samples x D dimensions embedding
        returns the flattened N(N-1)/2 vector
        of the upper triangle of an N x N distance matrix
        of sample distances under similarity metric
        """
        params = np.triu(
            pairwise_distances(sample_reps, metric=self._similarity_metric), k=1
        ).flatten()
        return params[params != 0]
