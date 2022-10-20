from brainscore_language import metric_registry
from .metric import PearsonCorrelation

metric_registry['pearsonr'] = PearsonCorrelation
