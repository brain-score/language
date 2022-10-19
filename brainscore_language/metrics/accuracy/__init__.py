from brainscore_language import metric_registry
from .metric import Accuracy

metric_registry['accuracy'] = Accuracy
