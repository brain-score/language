import typing

import xarray as xr


class Dataset:
    def __init__(self, xr_obj: xr.DataArray) -> "Dataset":
        """
        accepts an xarray with the following core
        dimensions: sampleid, neuroid, timeid
        and at least the following core
        coordinates: sampleid, neuroid, timeid, stimulus, subject

        Args:
            xr_obj (xr.DataArray): xarray object with core dimensions and coordinates
        """
        dims = ("sampleid", "neuroid", "timeid")
        coords = dims + ("stimulus", "subject")
        assert xr_obj.ndim == len(dims)
        assert all([dim in xr_obj.dims for dim in dims])
        assert all([coord in xr_obj.coords for coord in coords])
        self._xr_obj = xr_obj

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
