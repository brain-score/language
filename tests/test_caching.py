
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


def cacheable__eq__(o1: _Cacheable, o2: _Cacheable) -> bool:
        
        def checkattr(key) -> bool:
            try:
                if getattr(o1, key) != getattr(o2, key):
                    return False
            except AttributeError:
                return False
            return True

        for key, ob in vars(o1).items():
            if isinstance(ob, (str, Number, bool, _Cacheable, type(None))):
                if not checkattr(key): 
                    return False
            elif isinstance(ob, xr.DataArray):
                if not getattr(o1, key).identical(getattr(o2, key)):
                    return False
        else:
            return True

def test_cache_dataset() -> bool:

    p = Path('./.cache').expanduser().resolve()
    log(f'caching initialized dataset {dataset} to {p}')
    dataset.to_cache(cache_dir=p)

    dataset2 = lbs.dataset.Dataset.from_cache(identifier_string='<Dataset#dataset_name=Pereira2018NatStories>',
                                              cache_dir=p)

    return cacheable__eq__(dataset, dataset2)

if __name__ == '__main__':
    result = test_cache_dataset()
    print(result)