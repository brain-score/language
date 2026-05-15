from brainscore_language import benchmark_registry
from .benchmark import Fedorenko2016_ridge, Fedorenko2016_linear_shuffle

benchmark_registry['Fedorenko2016-ridge'] = Fedorenko2016_ridge
benchmark_registry['Fedorenko2016-linear-shuffle'] = Fedorenko2016_linear_shuffle