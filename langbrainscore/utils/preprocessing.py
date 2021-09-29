from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    KBinsDiscretizer,
    RobustScaler,
)
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA

# TODO: decide whether hardocded n_bins, n_components should be kept that way
#   or allowed to be changed
preprocessor_classes = {
    "demean": StandardScaler(with_std=False),
    "demean_std": StandardScaler(with_std=True),
    "minmax": MinMaxScaler,
    "discretize": KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform"),
    "robust_scaler": RobustScaler(),
    "pca": PCA(n_components=10),
    "gaussian_random_projection": GaussianRandomProjection(n_components=10),
}
