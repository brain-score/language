from brainscore_language import metric_registry
from .metric import CKACrossValidated

metric_registry['cka'] = CKACrossValidated