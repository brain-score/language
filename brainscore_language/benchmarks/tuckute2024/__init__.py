from brainscore_language import benchmark_registry
from .benchmark import Tuckute2024_linear, Tuckute2024_ridge, Tuckute2024_rdm, Tuckute2024_cka

benchmark_registry["Tuckute2024-linear"] = Tuckute2024_linear
benchmark_registry["Tuckute2024-ridge"] = Tuckute2024_ridge
benchmark_registry["Tuckute2024-rdm"] = Tuckute2024_rdm
benchmark_registry["Tuckute2024-cka"] = Tuckute2024_cka