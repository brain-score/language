"""submodule housing metrics used to evaluate similarity of representations"""

import typing
from langbrainscore.metrics.metric import (
    CKA,
    FisherCorr,
    PearsonR,
    RSA,
    ClassificationAccuracy,
    RMSE,
    KendallTau,
    SpearmanRho,
    # Metric,
)

metric_classes = {
    "cka": CKA,
    "rsa": RSA,
    "pearsonr": PearsonR,
    "spearmanrho": SpearmanRho,
    "fishercorr": FisherCorr,
    "kendalltau": KendallTau,
    "rmse": RMSE,
    "acc": ClassificationAccuracy,
}


# def load_metric(metric_class: typing.Union[str, Metric]):
#     if metric_class in metric_class_mapping:
#         return Metric(metric_class_mapping[metric_class])
#     elif callable(metric_class):
#         return Metric(metric_class)
