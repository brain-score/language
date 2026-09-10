from brainscore_language import benchmark_registry
from .benchmark import Pereira2018_243sentences, Pereira2018_384sentences
from .benchmark import Pereira2018_243sentences_ridge, Pereira2018_384sentences_ridge
from .benchmark import Pereira2018_243sentences_linear_shuffle, Pereira2018_384sentences_linear_shuffle
from .unified import (Pereira2018_243sentences_unified, Pereira2018_384sentences_unified,
                      Pereira2018_243sentences_ridge_unified,
                      Pereira2018_384sentences_ridge_unified)

benchmark_registry['Pereira2018.243sentences-linear'] = Pereira2018_243sentences
benchmark_registry['Pereira2018.384sentences-linear'] = Pereira2018_384sentences
benchmark_registry['Pereira2018.243sentences-linear-unified'] = Pereira2018_243sentences_unified
benchmark_registry['Pereira2018.384sentences-linear-unified'] = Pereira2018_384sentences_unified

benchmark_registry['Pereira2018.243sentences-ridge'] = Pereira2018_243sentences_ridge
benchmark_registry['Pereira2018.384sentences-ridge'] = Pereira2018_384sentences_ridge

benchmark_registry['Pereira2018.243sentences-linear-shuffle'] = Pereira2018_243sentences_linear_shuffle
benchmark_registry['Pereira2018.384sentences-linear-shuffle'] = Pereira2018_384sentences_linear_shuffle

# Mirror what upstream registers. The ridge variants group cross-validation by
# story; the linear ones above split sentences at random, which leaks within a
# passage, and upstream retired their non-unified counterparts for that reason.
benchmark_registry['Pereira2018.243sentences-ridge-unified'] = Pereira2018_243sentences_ridge_unified
benchmark_registry['Pereira2018.384sentences-ridge-unified'] = Pereira2018_384sentences_ridge_unified
