from brainscore_language import benchmark_registry
from .benchmark import Fedorenko2016_ridge, Fedorenko2016_linear

benchmark_registry['Fedorenko2016-linear'] = Fedorenko2016_linear
benchmark_registry['Fedorenko2016-ridge'] = Fedorenko2016_ridge