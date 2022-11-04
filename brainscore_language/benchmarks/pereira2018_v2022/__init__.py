from .benchmark import Pereira2018_243sentences, Pereira2018_384sentences
from brainscore_language import benchmark_registry

benchmark_registry["Pereira2018_v2022.243sentences-linreg_pearsonr"] = Pereira2018_243sentences
benchmark_registry["Pereira2018_v2022.384sentences-linreg_pearsonr"] = Pereira2018_384sentences
