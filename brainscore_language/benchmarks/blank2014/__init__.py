from brainscore_language import benchmark_registry
from .benchmark import Blank2014_ridge, Blank2014_linear_shuffle

benchmark_registry['Blank2014-ridge'] = Blank2014_ridge
benchmark_registry['Blank2014-linear-shuffle'] = Blank2014_linear_shuffle
