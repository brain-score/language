# stdlib imports 
import typing
from pathlib import Path
from collections import abc

# installed package imports
import xarray as xr
import numpy as np
from tqdm import tqdm

# local imports
from langbrainscore.interface import _Dataset
from langbrainscore.utils.logging import log


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


    def to_netcdf(self, filename):
        '''
        outputs member xarray object to a netCDF file
        identified by `filename`. if it already exists,
        silently overwrites it.
        '''
        self._xr_obj.to_netcdf(filename)

    @classmethod
    def read_netcdf(cls, filename):
        '''
        loads a netCDF object from a file identified by `filename`.
        '''
        return cls(xr.load_dataarray(filename))


    @classmethod
    def from_file_or_url(cls, 
                         file_path_or_url: typing.Union[str, Path], 
                         # arguments related to constructing coordinates and coordinate
                         # dimensions in the xarray
                         data_column: str, 
                         # sampleid uniquely identifies a stimulus shown to a participant
                         # neuroid uniquely identifies 
                         sampleid_index: str, neuroid_index: str, timeid_index: str = None,
                         # column to use as the subject/participant identifier
                         # if None, entire data is assumed to be a single subject
                         # the significance of the subject is that neuroids are not shared
                         # across subjects; a particular neuroid 'x' 
                         subject_index: str = None,
                         stimuli_index: str = None,
                         # arguments related to non-dimension coordinates to track
                         # metadata
                         sampleid_metadata: typing.Union[typing.Iterable[str], typing.Mapping[str,str]] = None,
                         neuroid_metadata: typing.Union[typing.Iterable[str], typing.Mapping[str,str]] = None,
                         timeid_metadata: typing.Union[typing.Iterable[str], typing.Mapping[str,str]] = None,
                         #
                         multidim_metadata: typing.Iterable[typing.Mapping[str, typing.Iterable[str]]] = None,
                         #
                         sort_by: typing.Iterable[str] = (),
                         sep=',') -> _Dataset:
        """Constructs and returns a Dataset object based on data provided in a csv file.
            Constructs an xarray using specified columns to construct dimensions and 
            metadata along those dimensions.
            Minimally requires `sampleid` and `neuroid` to be provided.
            If `timeid` and `sampleid` is not provided:
                - a singleton timeid dimension is created with the value "0" for each sample.
                - a singleton subjectid dimension is created with value "0" that spans the entire data.
            
            For help on what these terms mean, please visit the
            [xarray glossary page](https://xarray.pydata.org/en/stable/user-guide/terminology.html)

            NOTE: Each row of the supplied file must have a single data point corresponding
            to a unique `sampleid`, `neuroid`, and `timeid` (unique dimension values). 
            I.e., each neuroid (which could be a voxel, an ROI, a reaction time RT value, etc.)
            must be on a new line for the same stimulus trial at a certain time. 

        Args:
            file_path_or_url (typing.Union[str, Path]): filepath or URL to file containing a CSV
            
            sampleid_column (str): Title of the column to be used as `sampleid` dimension coordinate.
            neuroid_column (str): Title of the column to be used as `neuroid` dimension coordinate. 
            timeid_column (str, optional): Title of the column to be used as `timeid` dimension coordinate. 
                If None is provided, a singleton timeid column with data at the 0th index will be created. 
                Defaults to None.
                
            sampleid_metadata (typing.Union[typing.Iterable, typing.Mapping][str], optional): 
                Column names (and optionally a corresponding name to rename the coordinate to) 
                that should be used as metadata along the `sampleid` dimension. 
            neuroid_metadata (typing.Union[typing.Iterable, typing.Mapping][str], optional)
                See description for `sampleid_metadata`, but with `neuroid`.
            timeid_metadata (typing.Union[typing.Iterable, typing.Mapping][str], optional)
                See description for `sampleid_metadata` but with `timeid`.

            multidim_metadata (typing.Iterable[typing.Mapping[str, typing.Iterable[str]]])

            sep (str, optional): separator string in the CSV/TSV file. Defaults to ','.
        """

        T = typing.TypeVar("T")
        def collapse_same_value(arr: typing.Iterable[T]) -> T:
            '''
            makes sure each element in an iterable is identical (using __eq__)
            to every other element by value and returns (any) one of the elements.
            if a non-identical element (!=) is found, raises ValueError
            '''
            first_thing = next(iter(arr))
            for each_thing in arr:
                if first_thing != each_thing:
                    raise ValueError(f'{first_thing} != {each_thing}')
            return first_thing

        import pandas as pd
        df = pd.read_csv(file_path_or_url, sep=sep)

        if timeid_index is None:
            timeid_index = 'timeid'
            # create singleton timeid
            # we don't need to inflate data since each datapoint will just 
            # correspond to timeid == 0 per sample
            timeid_column = [0] * len(df)
            df[timeid_index] = timeid_column

        subjects = list(set(df[subject_index]))
        sampleids = list(set(df[sampleid_index]))
        neuroids = list(set(df[neuroid_index]))
        timeids = list(set(df[timeid_index]))

        if not isinstance(sampleid_metadata, abc.Mapping):
            sampleid_metadata = {k: k for k in sampleid_metadata}
        if not isinstance(neuroid_metadata, abc.Mapping):
            neuroid_metadata = {k: k for k in neuroid_metadata}
        if not isinstance(timeid_metadata, abc.Mapping):
            timeid_metadata = {k: k for k in timeid_metadata or ()}

        df = df.sort_values([subject_index, sampleid_index, neuroid_index, timeid_index])

        sampleid_xrs = []
        for sampleid in tqdm(sampleids, desc='reassembling data per sampleid'):
            sample_view = df[df[sampleid_index] == sampleid]

            neuroid_xrs = []
            for neuroid in neuroids:
                neuroid_view = sample_view[sample_view[neuroid_index] == neuroid]

                timeid_xrs = []
                for timeid in timeids:
                    timeid_view = neuroid_view[neuroid_view[timeid_index] == timeid]
                    data = timeid_view[data_column].values
                    timeid_xr = xr.DataArray(
                            data.reshape(1,len(timeid_view[subject_index]),1),
                            dims=("sampleid", "neuroid", "timeid"),
                            coords={
                                "sampleid": np.repeat(sampleid, 1),
                                "neuroid": [f'{a}_{b}' for a, b in zip(timeid_view[subject_index], timeid_view[neuroid_index])],
                                "timeid": np.repeat(timeid, 1),
                                "subject": ('neuroid', timeid_view[subject_index]),
                                "stimulus": ('sampleid', [collapse_same_value(timeid_view[stimuli_index])]),
                                **{metadata_names[column]: (dimension, [collapse_same_value(timeid_view[column])])
                                   for dimension, metadata_names in (('sampleid', sampleid_metadata), 
                                                                     ('timeid', timeid_metadata))
                                   for column in metadata_names
                                },
                                **{neuroid_metadata[column]: ('neuroid', (timeid_view[column]))
                                   for column in neuroid_metadata
                                }
                            }
                        )
                    timeid_xrs += [timeid_xr]

                neuroid_xr = xr.concat(timeid_xrs, dim='timeid')
                neuroid_xrs += [neuroid_xr]
            
            sampleid_xr = xr.concat(neuroid_xrs, dim='neuroid')
            sampleid_xrs += [sampleid_xr]

        unified_xr = xr.concat(sampleid_xrs, dim='sampleid')
        
        from langbrainscore.utils.xarray import collapse_multidim_coord
        for dimension, metadata_names in (('sampleid', sampleid_metadata + {'stimulus':'stimulus'}), 
                                          ('timeid', timeid_metadata),
                                          ('neuroid', neuroid_metadata + {'subject': 'subject'})):
            for column in metadata_names:
                try:
                    unified_xr = collapse_multidim_coord(unified_xr, metadata_names[column], dimension)
                except ValueError as e:
                    log(f'dimension:{dimension}, column:{column}, shape:{unified_xr[metadata_names[column]].shape}', type='ERR')

        return cls(unified_xr) # NOTE: we use `cls` rather than `Dataset` so any
                               # subclasses will use the subclass rather than parent  
        # ^ the above method addresses the below TODO.
        # TODO: we should adapt the above to package an xarray object automatically if a path to files is passed.
        # This functionality can be wrapped in a utility that we simply import and utilize here.
 