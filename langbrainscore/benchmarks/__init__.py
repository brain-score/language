import typing
from pathlib import Path
from langbrainscore.benchmarks.pereira2018 import pereira2018_mean_froi_nat_stories


supported_benchmarks: typing.Mapping[str, typing.Callable] = {
    "pereira2018_mean_froi_nat_stories": pereira2018_mean_froi_nat_stories,
    "pereira2018_ind_voxels_nat_stories": NotImplemented,
}


def load_benchmark(
    benchmark_name_or_path: typing.Union[str, Path], *loading_args, **loading_kwargs
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
        return loader(*loading_args, **loading_kwargs)
    raise NotImplementedError(
        f"Benchmark identified by `{benchmark_name_or_path}` currently not supported."
    )
