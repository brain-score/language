from brainscore_language import benchmark_registry
from .benchmark import Pereira2018_243sentences_ridge, Pereira2018_384sentences_ridge
from .benchmark import Pereira2018_243sentences_linear, Pereira2018_384sentences_linear

benchmark_registry['Pereira2018.243sentences-ridge'] = Pereira2018_243sentences_ridge
benchmark_registry['Pereira2018.384sentences-ridge'] = Pereira2018_384sentences_ridge

benchmark_registry['Pereira2018.243sentences-linear'] = Pereira2018_243sentences_linear
benchmark_registry['Pereira2018.384sentences-linear'] = Pereira2018_384sentences_linear