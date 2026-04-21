from brainscore_language import benchmark_registry
from .benchmark import Pereira2018_243sentences, Pereira2018_384sentences
from .unified import Pereira2018_243sentences_unified, Pereira2018_384sentences_unified

benchmark_registry['Pereira2018.243sentences-linear'] = Pereira2018_243sentences
benchmark_registry['Pereira2018.384sentences-linear'] = Pereira2018_384sentences
benchmark_registry['Pereira2018.243sentences-linear-unified'] = Pereira2018_243sentences_unified
benchmark_registry['Pereira2018.384sentences-linear-unified'] = Pereira2018_384sentences_unified
