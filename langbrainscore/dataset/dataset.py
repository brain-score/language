import typing

import xarray as xr
from langbrainscore.interface import _Dataset


class Dataset(_Dataset):
    def __init__(self, xr_obj: xr.DataArray) -> "Dataset":
        super().__init__(xr_obj)

    @property
    def contents(self) -> xr.DataArray:
        """
        access the internal xarray object. use with caution.
        """
        return self._xr_obj

    @property
    def stimuli(self) -> xr.DataArray:
        """
        getter method that returns an xarray object of stimuli and associated metadata

        Returns:
            xr.DataArray: xarray object containing the stimuli from the dataset and associated metadata
        """
        return self.contents.stimulus

    @property
    def dims(self) -> tuple:
        """
        getter method that returns internal xarray dimensions

        Returns:
            tuple[str]: dimensions of internal xarray object
        """
        return self.contents.dims


# TODO: we should adapt the above to package an xarray object automatically if a path to files is passed.
# This functionality can be wrapped in a utility that we simply import and utilize here.
