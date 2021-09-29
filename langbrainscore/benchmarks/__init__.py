import typing
from pathlib import Path
from functools import partial

from langbrainscore.benchmarks.pereira2018 import pereira2018_mean_froi


supported_benchmarks: typing.Mapping[str, typing.Callable] = {
    "pereira2018_mean_froi_Lang": partial(pereira2018_mean_froi, network="Lang"),
    "pereira2018_mean_froi_MD": partial(pereira2018_mean_froi, network="MD"),
    "pereira2018_mean_froi": pereira2018_mean_froi,
    "pereira2018_ind_voxels": NotImplemented,
}


def load_benchmark(
    benchmark_name_or_path: typing.Union[str, Path],
    *loading_args,
    load_cache=True,
    **loading_kwargs,
) -> "langbrainscore.dataset.Dataset":
    """A method that, given a name or a path to a benchmark, loads it and returns
        a dataset object.

    Args:
        benchmark_name_or_path (typing.Union[str, Path]): name of a pre-packaged/officially supported benchmark
            or a path to a benchmark with valid formatting

    Returns:
        langbrainscore.dataset.Dataset: a langbrainscore Dataset object
    """
    if benchmark_name_or_path in supported_benchmarks:
        loader = supported_benchmarks[benchmark_name_or_path]
        return loader(*loading_args, load_cache=load_cache, **loading_kwargs)
    raise NotImplementedError(
        f"Benchmark identified by `{benchmark_name_or_path}` currently not supported."
    )
