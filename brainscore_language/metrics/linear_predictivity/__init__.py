from brainscore_language import metric_registry
from .metric import linear_pearsonr, ridge_pearsonr

metric_registry['linear_pearsonr'] = linear_pearsonr
metric_registry['ridge_pearsonr'] = ridge_pearsonr