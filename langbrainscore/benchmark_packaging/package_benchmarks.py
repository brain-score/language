from pathlib import Path

import pandas as pd
import xarray as xr
from langbrainscore.utils import logging


def cache_dataset(loader, cache_dir=None):
    def wrapper(key, data_dir):
        fname = f"{key}.nc"
        try:
            cache_dir = cache_dir or os.environ["LANGBRAINSCORE_CACHE"]
        except KeyError:
            cache_dir = Path(__file__).parent.parent.resolve()
            os.environ["LANGBRAINSCORE_CACHE"] = str(cache_dir)
            logging.log(f"Set cache directory to {cache_dir}")
        file = cache_dir / fname
        if file.exists():
            dataset = xr.open_dataset(str(file))
            return dataset
        else:
            dataset = loader(key, data_dir)
            dataset.to_netcdf(str(file))
            return dataset

    return wrapper


@cache_dataset
def load_dataset(key, data_dir=Path("../data")):
    # key = firstauthor_pubyear_datatype_identifierstring
    datatype = key.split("_")[2]
    assert datatype in ["imaze", "blockedfMRI", "eventfMRI", "MEGtimeseries"]
    loader = globals()[f"load_{datatype}"]
    return loader(key, data_dir)


def load_imaze(key, data_dir):
    data = pd.read_csv(str(datadir / key) + ".csv")
    raise NotImplementedError("resume here")


def load_blockedfMRI(key, data_dir):
    pass


def load_eventfRMI(key, data_dir):
    pass
