
from pathlib import Path
import typing
from numbers import Number
from langbrainscore.interface.cacheable import _Cacheable
import xarray as xr
import langbrainscore as lbs
import langbrainscore.benchmarks
from langbrainscore.utils.logging import log

pereira_xr = lbs.benchmarks.pereira2018_nat_stories()
dataset = lbs.dataset.Dataset(pereira_xr)


def test_cache_dataset() -> bool:

    p = Path('./.cache').expanduser().resolve()
    log(f'caching initialized dataset {dataset} to {p}')
    dataset.to_cache(cache_dir=p)

    dataset2 = lbs.dataset.Dataset.from_cache(identifier_string='<Dataset#dataset_name=Pereira2018NatStories>',
                                              cache_dir=p)

    return dataset == dataset2

if __name__ == '__main__':
    result = test_cache_dataset()
    print(result)