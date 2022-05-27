# stdlib imports
import typing
from collections import abc
from pathlib import Path

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from langbrainscore.interface import _Dataset
from langbrainscore.utils.logging import log
from langbrainscore.utils.xarray import collapse_multidim_coord
from tqdm import tqdm


class Dataset(_Dataset):
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

    def to_netcdf(self, filename):
        """
        outputs the xarray.DataArray object to a netCDF file identified by
        `filename`. if it already exists, overwrites it.
        """
        if Path(filename).expanduser().resolve().exists():
            log(f"{filename} already exists. overwriting.", type="WARN")
        self._xr_obj.to_netcdf(filename)

    @classmethod
    def load_netcdf(cls, filename):
        """
        loads a netCDF object that contains a pre-packaged xarray instance from
        a file at `filename`.
        """
        return cls(xr.load_dataarray(filename))

    @classmethod
    def from_file_or_url(
        cls,
        file_path_or_url: typing.Union[str, Path],
        data_column: str,
        sampleid_index: str,
        neuroid_index: str,
        stimuli_index: str,
        timeid_index: str = None,
        subject_index: str = None,
        sampleid_metadata: typing.Union[
            typing.Iterable[str], typing.Mapping[str, str]
        ] = None,
        neuroid_metadata: typing.Union[
            typing.Iterable[str], typing.Mapping[str, str]
        ] = None,
        timeid_metadata: typing.Union[
            typing.Iterable[str], typing.Mapping[str, str]
        ] = None,
        multidim_metadata: typing.Iterable[
            typing.Mapping[str, typing.Iterable[str]]
        ] = None,
        sort_by: typing.Iterable[str] = (),
        sep=",",
        parallel: int = -2,
    ) -> _Dataset:
        """Creates a Dataset object holding an `xr.DataArray` instance using a CSV file readable by pandas.
            Constructs the `xr.DataArray` using specified columns to construct dimensions and
            metadata along those dimensions in the form of coordinates.
            Minimally requires `sampleid` and `neuroid` to be provided.

            Note: Each row of the supplied file must have a single data point corresponding
            to a unique `sampleid`, `neuroid`, and `timeid` (unique dimension values).
            I.e., each neuroid (which could be a voxel, an ROI, a reaction time RT value, etc.)
            must be on a new line for the same stimulus trial at a certain time.
            If `timeid` and `subjectid` is not provided:
                - a singleton timeid dimension is created with the value "0" for each sample.
                - a singleton subjectid dimension is created with value "0" that spans the entire data.
            For help on what these terms mean, please visit the
            [xarray glossary page](https://xarray.pydata.org/en/stable/user-guide/terminology.html)


        Args:
            file_path_or_url (typing.Union[str, Path]): a path or URL to a csv file
            data_column (str): title of the column that holds the datapoints per unit of measurement
                (e.g., BOLD contrast effect size, reaction time, voltage amplitude, etc)
            sampleid_index (str): title of the column that should be used to construct an index for sampleids.
                this should be unique for each stimulus in the dataset.
            neuroid_index (str): title of the column that should be used to construct an index for neuroids.
                this should be unique for each point of measurement within a subject. e.g., voxel1, voxel2, ...
                neuroids in the packaged dataset are transformed to be a product of subject_index and neuroid_index.
            stimuli_index (str): title of the column that holds stimuli shown to participants
            timeid_index (str, optional): title of the column that holds timepoints of stimulus presentation.
                optional. if not provided, a singleton timepoint '0' is assigned to each datapoint. Defaults to None.
            subject_index (str, optional): title of the column specifiying subject IDs. Defaults to None.
            sampleid_metadata (typing.Union[typing.Iterable[str], typing.Mapping[str,str]], optional):
                names of columns (and optionally mapping of existing column names to new coordinate names)
                that supply metadata along the sampleid dimension. Defaults to None.
            neuroid_metadata (typing.Union[typing.Iterable[str], typing.Mapping[str,str]], optional):
                see `sampleid_metadata`. Defaults to None.
            timeid_metadata (typing.Union[typing.Iterable[str], typing.Mapping[str,str]], optional):
                see `sampleid_metadata`. Defaults to None.
            multidim_metadata (typing.Iterable[typing.Mapping[str, typing.Iterable[str]]], optional):
                metadata to go with more than one dimension. e.g., chunks of a stimulus that unfolds with time.
                currently `NotImplemented`. Defaults to None.
            sort_by (typing.Iterable[str], optional): Sort data by these columns while repackaging it.
                data is sorted by `sampleid_index`, `neuroid_index`, and `timeid_index` in addition to this
                value. Defaults to ().
            sep (str, optional): separator to read a value-delimited file. this argument is passed to pandas.
                Defaults to ','.

        Raises:
            ValueError: _description_

        Returns:
            _Dataset: a subclass of the `_Dataset` interface with the packaged xarray.DataArray as a member.
        """

        T = typing.TypeVar("T")

        def collapse_same_value(arr: typing.Iterable[T]) -> T:
            """
            makes sure each element in an iterable is identical (using __eq__)
            to every other element by value and returns (any) one of the elements.
            if a non-identical element (!=) is found, raises ValueError
            """
            try:
                first_thing = next(iter(arr))
            except StopIteration:
                log(f"failed to obtain value from {arr}", verbosity_check=True)
                return np.nan
            for each_thing in arr:
                if first_thing != each_thing:
                    raise ValueError(f"{first_thing} != {each_thing}")
            return first_thing

        import pandas as pd

        if str(file_path_or_url).endswith(".parquet.gzip"):
            try:
                df = pd.read_parquet(file_path_or_url)
            except Exception as invalid_file:
                raise ValueError("invalid parquet file / filename") from invalid_file
        else:
            try:
                df = pd.read_csv(file_path_or_url, sep=sep)
            except Exception as invalid_file:
                raise ValueError("invalid csv file / filename") from invalid_file

        if timeid_index is None:
            timeid_index = "timeid"
            # create singleton timeid
            # we don't need to inflate data since each datapoint will just
            # correspond to timeid == 0 per sample
            timeid_column = [0] * len(df)
            df[timeid_index] = timeid_column
        if subject_index is None:
            subject_index = "subject"
            # create singleton subjectID
            # we don't need to inflate data since each datapoint will just
            # correspond to subject == 0 per sample
            subject_column = ["subject0"] * len(df)
            df[subject_index] = subject_column
        if not parallel:
            parallel = 1

        subjects = list(set(df[subject_index]))
        sampleids = list(
            set(df[sampleid_index])
        )  # what happens when the same stimulus is shown multiple times?
        # it will add entries with same sampleid that will have to
        # then be differentiated on the basis of metadata only
        # https://i.imgur.com/4V2DsIo.png
        neuroids = list(set(df[neuroid_index]))
        timeids = list(set(df[timeid_index]))

        if not isinstance(sampleid_metadata, abc.Mapping):
            sampleid_metadata = {k: k for k in sampleid_metadata}
        if not isinstance(neuroid_metadata, abc.Mapping):
            neuroid_metadata = {k: k for k in neuroid_metadata}
        if not isinstance(timeid_metadata, abc.Mapping):
            timeid_metadata = {k: k for k in timeid_metadata or ()}

        df = df.sort_values(
            [*sort_by, subject_index, sampleid_index, neuroid_index, timeid_index]
        )

        def get_sampleid_xr(sampleid):
            sample_view = df[df[sampleid_index] == sampleid]  # why not sampleid_view?

            neuroid_xrs = []
            for neuroid in neuroids:
                neuroid_view = sample_view[sample_view[neuroid_index] == neuroid]

                timeid_xrs = []
                for timeid in timeids:
                    timeid_view = neuroid_view[neuroid_view[timeid_index] == timeid]
                    data = timeid_view[data_column].values
                    timeid_xr = xr.DataArray(
                        data.reshape(1, len(timeid_view[subject_index]), 1),
                        dims=("sampleid", "neuroid", "timeid"),
                        coords={
                            "sampleid": np.repeat(sampleid, 1),
                            "neuroid": [
                                f"{a}_{b}"
                                for a, b in zip(
                                    timeid_view[subject_index],
                                    timeid_view[neuroid_index],
                                )
                            ],
                            "timeid": np.repeat(timeid, 1),
                            "subject": ("neuroid", timeid_view[subject_index]),
                            "stimulus": (
                                "sampleid",
                                [collapse_same_value(timeid_view[stimuli_index])],
                            ),
                            **{
                                metadata_names[column]: (
                                    dimension,
                                    [collapse_same_value(timeid_view[column])],
                                )
                                for dimension, metadata_names in (
                                    ("sampleid", sampleid_metadata),
                                    ("timeid", timeid_metadata),
                                )
                                for column in metadata_names
                            },
                            **{
                                neuroid_metadata[column]: (
                                    "neuroid",
                                    (timeid_view[column]),
                                )
                                for column in neuroid_metadata
                            },
                        },
                    )
                    timeid_xrs += [timeid_xr]

                neuroid_xr = xr.concat(timeid_xrs, dim="timeid")
                neuroid_xrs += [neuroid_xr]

            sampleid_xr = xr.concat(neuroid_xrs, dim="neuroid")
            return sampleid_xr

        sampleid_xrs = Parallel(n_jobs=parallel)(
            delayed(get_sampleid_xr)(sampleid)
            for sampleid in tqdm(sampleids, desc="reassembling data per sampleid")
        )

        unified_xr = xr.concat(sampleid_xrs, dim="sampleid")

        for dimension, metadata_names in (
            ("sampleid", {**sampleid_metadata, "stimulus": "stimulus"}),
            ("timeid", timeid_metadata),
            ("neuroid", {**neuroid_metadata, "subject": "subject"}),
        ):
            for column in metadata_names:
                try:
                    unified_xr = collapse_multidim_coord(
                        unified_xr, metadata_names[column], dimension
                    )
                except ValueError as e:
                    log(
                        f"dimension:{dimension}, column:{column}, shape:{unified_xr[metadata_names[column]].shape}",
                        type="ERR",
                    )

        return cls(unified_xr)  # NOTE: we use `cls` rather than `Dataset` so any
        # subclasses will use the subclass rather than parent
