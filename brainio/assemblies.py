import logging
from collections import OrderedDict, defaultdict

import itertools

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray, IndexVariable

_logger = logging.getLogger(__name__)


def is_fastpath(*args, **kwargs):
    """checks whether a set of args and kwargs would be interpreted by DataArray.__init__"""
    n = 7 # maximum length of args if all arguments to DataArray are positional (as of 0.16.1)
    return ("fastpath" in kwargs and kwargs["fastpath"]) or (len(args) >= n and args[n-1])


class DataPoint(object):
    """A DataPoint represents one value, usually a recording from one neuron or node,
    in response to one presentation of a stimulus.  """

    def __init__(self, value, neuroid, presentation):
        self.value = value
        self.neuroid = neuroid
        self.presentation = presentation


class DataAssembly(DataArray):
    """A DataAssembly represents a set of data a researcher wishes to work with for
    an analysis or benchmarking task.  """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        if is_fastpath(*args, **kwargs):
            # DataArray.__init__ follows a very different code path if fastpath=True
            # gather_indexes is not necessary in those cases
            super(DataAssembly, self).__init__(*args, **kwargs)
        else:
            # We call gather_indexes so we can guarantee that DataAssemblies will always have all metadata as indexes.
            # set_index, and thus gather_indexes, cannot operate on self in-place, it can only return a new object.
            # We take advantage of the almost-idempotence of DataArray.__init__ to gather indexes on a temporary
            # object and then initialize self with that.
            temp = DataArray(*args, **kwargs)
            temp = gather_indexes(temp)
            super(DataAssembly, self).__init__(temp)

    @classmethod
    def get_loader_class(cls):
        return StimulusMergeAssemblyLoader

    @classmethod
    def from_files(cls, file_path, **kwargs):
        loader_class = cls.get_loader_class()
        loader = loader_class(
            cls=cls,
            file_path=file_path,
            **kwargs,
        )
        return loader.load()

    def validate(self):
        pass

    def multi_groupby(self, group_coord_names, *args, **kwargs):
        if len(group_coord_names) < 2:
            return self.groupby(group_coord_names[0], *args, **kwargs)
        multi_group_name = "multi_group"
        dim = self._dim_of_group_coords(group_coord_names)
        tmp_assy = self._join_group_coords(dim, group_coord_names, multi_group_name)
        result = tmp_assy.groupby(multi_group_name, *args, **kwargs)
        return GroupbyBridge(result, self, dim, group_coord_names, multi_group_name)

    def _join_group_coords(self, dim, group_coord_names, multi_group_name):
        class MultiCoord:
            # this is basically a list of key-values, but not treated as a list to avoid xarray complaints
            def __init__(self, values):
                self.values = tuple(values) if isinstance(values, list) else values

            def __eq__(self, other):
                return len(self.values) == len(other.values) and \
                       all(v1 == v2 for v1, v2 in zip(self.values, other.values))

            def __lt__(self, other):
                return self.values < other.values

            def __hash__(self):
                return hash(self.values)

            def __repr__(self):
                return repr(self.values)

        tmp_assy = self.copy()
        group_coords = [tmp_assy.coords[c].values.tolist() for c in group_coord_names]
        multi_group_coord = []
        for coords in zip(*group_coords):
            multi_group_coord.append(MultiCoord(coords))
        tmp_assy.coords[multi_group_name] = dim, multi_group_coord
        tmp_assy = tmp_assy.set_index(append=True, **{dim: multi_group_name})
        return tmp_assy

    def _dim_of_group_coords(self, group_coord_names):
        dimses = [self.coords[coord_name].dims for coord_name in group_coord_names]
        dims = [dim for dim_tuple in dimses for dim in dim_tuple]
        if len(set(dims)) == 1:
            return dims[0]
        else:
            raise GroupbyError("All coordinates for grouping must be associated with the same single dimension.  ")

    def multi_dim_apply(self, groups, apply):
        # align
        groups = sorted(groups, key=lambda group: self.dims.index(self[group].dims[0]))
        # build indices
        groups = {group: np.unique(self[group]) for group in groups}
        group_dims = {self[group].dims: group for group in groups}
        indices = defaultdict(lambda: defaultdict(list))
        result_indices = defaultdict(lambda: defaultdict(list))
        for group in groups:
            for index, value in enumerate(self[group].values):
                indices[group][value].append(index)
                # result_indices
                index = max(itertools.chain(*result_indices[group].values())) + 1 \
                    if len(result_indices[group]) > 0 else 0
                result_indices[group][value].append(index)

        coords = {coord: (dims, value) for coord, dims, value in walk_coords(self)}

        def simplify(value):
            return value.item() if value.size == 1 else value

        def indexify(dict_indices):
            return tuple((i,) if isinstance(i, int) else tuple(i) for i in dict_indices.values())

        # group and apply
        # making this a DataArray right away and then inserting through .loc would slow things down
        shapes = {group: len(list(itertools.chain(*indices.values()))) for group, indices in result_indices.items()}
        result = np.zeros(list(shapes.values()))
        result_coords = {coord: (dims, (np.array([None] * shapes[group_dims[dims]])
                                        # deal with non-indexing coords (typically remnants from `.sel(coord=x)`)
                                        if dims else value))
                         for coord, (dims, value) in coords.items()}
        for values in itertools.product(*groups.values()):
            group_values = dict(zip(groups.keys(), values))
            self_indices = {group: indices[group][value] for group, value in group_values.items()}
            values_indices = indexify(self_indices)
            cells = self.values[values_indices]  # using DataArray would slow things down. thus we pass coords as kwargs
            cells = simplify(cells)
            cell_coords = {coord: (dims,
                                   value[self_indices[group_dims[dims]]]
                                   if dims else value)  # deal with non-indexing coords
                           for coord, (dims, value) in coords.items()}
            cell_coords = {coord: (dims, simplify(value)) for coord, (dims, value) in cell_coords.items()}

            # ignore dims when passing to function
            passed_coords = {coord: value for coord, (dims, value) in cell_coords.items()}
            merge = apply(cells, **passed_coords)
            result_idx = {group: result_indices[group][value] for group, value in group_values.items()}
            result[indexify(result_idx)] = merge
            for coord, (dims, value) in cell_coords.items():
                assert dims == result_coords[coord][0]
                if not dims:  # non-indexing coord
                    continue
                coord_index = result_idx[group_dims[dims]]
                result_coords[coord][1][coord_index] = value

        # re-package
        result = type(self)(result, coords=result_coords, dims=list(itertools.chain(*group_dims.keys())))
        return result

    def multisel(self, method=None, tolerance=None, drop=False, **indexers):
        """
        partial workaround to keep multi-indexes and scalar coords
        https://github.com/pydata/xarray/issues/1491, https://github.com/pydata/xarray/pull/1426
        this method might slow things down, use with caution
        """
        indexer_dims = {index: self[index].dims for index in indexers}
        dims = []
        for _dims in indexer_dims.values():
            assert len(_dims) == 1
            dims.append(_dims[0])
        coords_dim, dim_coords = {}, defaultdict(list)
        for dim in dims:
            for coord, coord_dims, _ in walk_coords(self):
                if array_is_element(coord_dims, dim):
                    coords_dim[coord] = dim
                    dim_coords[dim].append(coord)

        result = super().sel(method=method, tolerance=tolerance, drop=drop, **indexers)

        # un-drop potentially dropped dims
        for coord, value in indexers.items():
            dim = self[coord].dims
            assert len(dim) == 1
            dim = dim[0]
            if not hasattr(result, coord) and dim not in result.dims:
                result = result.expand_dims(coord)
                result[coord] = [value]

        # stack back together
        stack_dims = list(result.dims)
        for result_dim in stack_dims:
            if result_dim not in self.dims:
                original_dim = coords_dim[result_dim]
                stack_coords = [coord for coord in dim_coords[original_dim] if hasattr(result, coord)]
                for coord in stack_coords:
                    stack_dims.remove(coord)
                result = result.stack(**{original_dim: stack_coords})
        # add scalar indexer variable
        for index, value in indexers.items():
            if hasattr(result, index):
                continue  # already set, potentially during un-dropping
            dim = indexer_dims[index]
            assert len(dim) == 1
            value = np.repeat(value, len(result[dim[0]]))
            result[index] = dim, value
        return result


class BehavioralAssembly(DataAssembly):
    """A BehavioralAssembly is a DataAssembly containing behavioral data.  """
    __slots__ = ()


class NeuroidAssembly(DataAssembly):
    """A NeuroidAssembly is a DataAssembly containing data recorded from either neurons
    or neuron analogues.  """
    __slots__ = ()

    def validate(self):
        assert set(self.dims) == {'presentation', 'neuroid'} or \
               set(self.dims) == {'presentation', 'neuroid', 'time_bin'}


class NeuronRecordingAssembly(NeuroidAssembly):
    """A NeuronRecordingAssembly is a NeuroidAssembly containing data recorded from neurons.  """
    __slots__ = ()


class ModelFeaturesAssembly(NeuroidAssembly):
    """A ModelFeaturesAssembly is a NeuroidAssembly containing data captured from nodes in
    a machine learning model.  """
    __slots__ = ()


class PropertyAssembly(DataAssembly):
    """A PropertyAssembly is a DataAssembly containing single neuronal properties data.  """
    __slots__ = ()

    @classmethod
    def get_loader_class(cls):
        return StimulusReferenceAssemblyLoader


class SpikeTimesAssembly(NeuronRecordingAssembly):
    """A SpikeTimesAssembly is a DataAssembly containing a one-dimensional array of neural spike event timestamps.  """
    __slots__ = ()

    @classmethod
    def get_loader_class(cls):
        return GroupAppendAssemblyLoader

    def validate(self):
        assert set(self.dims) == {'event'}


class MetadataAssembly(DataAssembly):
    """A MetadataAssembly is a DataAssembly containing metadata, pertaining to another DataAssembly, that is best
    described by a DataAssembly but has different dimensions from the DataAssembly it describes.  """
    __slots__ = ()

    @classmethod
    def get_loader_class(cls):
        return StimulusReferenceAssemblyLoader


class GroupbyBridge(object):
    """Wraps an xarray GroupBy object to allow grouping on multiple coordinates.   """

    def __init__(self, groupby, assembly, dim, group_coord_names, multi_group_name):
        self.groupby = groupby
        self.assembly = assembly
        self.dim = dim
        self.group_coord_names = group_coord_names
        self.multi_group_name = multi_group_name

    def __getattr__(self, attr):
        result = getattr(self.groupby, attr)
        if callable(result):
            result = self.wrap_groupby(result)
        return result

    def wrap_groupby(self, func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, type(self.assembly)):
                result = self.split_group_coords(result)
            return result

        return wrapper

    def split_group_coords(self, result):
        split_coords = [multi_coord.values for multi_coord in result.coords[self.multi_group_name].values]
        split_coords = list(map(list, zip(*split_coords)))  # transpose
        for coord_name, coord in zip(self.group_coord_names, split_coords):
            result.coords[coord_name] = (self.multi_group_name, coord)
        result_type = type(result)
        result = xr.DataArray(result).reset_index(self.multi_group_name, drop=True)
        result = result_type(result.set_index(append=True, **{self.multi_group_name: list(self.group_coord_names)}))
        result = result.rename({self.multi_group_name: self.dim})
        return result


class GroupbyError(Exception):
    pass


def merge_data_arrays(data_arrays):
    # https://stackoverflow.com/a/50125997/2225200
    merged = xr.merge((data_array.rename('z') for data_array in data_arrays))['z'].rename(None)
    # ensure same class
    return type(data_arrays[0])(merged)


def array_is_element(arr, element):
    return len(arr) == 1 and arr[0] == element


def get_metadata(assembly, dims=None, names_only=False, include_coords=True,
                 include_indexes=True, include_multi_indexes=False, include_levels=True):
    """
    Return coords and/or indexes or index levels from an assembly, yielding either `name` or `(name, dims, values)`.
    """
    def what(name, dims, values, names_only):
        if names_only:
            return name
        else:
            return name, dims, values
    if dims is None:
        dims = assembly.dims + (None,) # all dims plus dimensionless coords
    for name in assembly.coords.variables:
        values = assembly.coords.variables[name]
        is_subset = values.dims and (set(values.dims) <= set(dims))
        is_dimless = (not values.dims) and None in dims
        if is_subset or is_dimless:
            is_index = isinstance(values, IndexVariable)
            if is_index:
                if values.level_names: # it's a MultiIndex
                    if include_multi_indexes:
                        yield what(name, values.dims, values.values, names_only)
                    if include_levels:
                        for level in values.level_names:
                            level_values = assembly.coords[level]
                            yield what(level, level_values.dims, level_values.values, names_only)
                else: # it's an Index
                    if include_indexes:
                        yield what(name, values.dims, values.values, names_only)
            else:
                if include_coords:
                    yield what(name, values.dims, values.values, names_only)


def coords_for_dim(assembly, dim):
    result = OrderedDict()
    meta = get_metadata(assembly, dims=(dim,), include_indexes=False, include_levels=False)
    for name, dims, values in meta:
        result[name] = values
    return result


def walk_coords(assembly):
    """
    walks through coords and all levels, just like the `__repr__` function, yielding `(name, dims, values)`.
    """
    yield from get_metadata(assembly)


def get_levels(assembly):
    levels = list(get_metadata(assembly, names_only=True, include_coords=False, include_indexes=False))
    return levels


def gather_indexes(assembly):
    """This is only necessary as long as xarray cannot persist MultiIndex to netCDF.  """
    coords_d = {}
    for dim in assembly.dims:
        coord_names = list(get_metadata(assembly, dims=(dim,), names_only=True, include_indexes=False, include_levels=False))
        if coord_names:
            coords_d[dim] = coord_names
    if coords_d:
        assembly = assembly.set_index(append=True, **coords_d)
    return assembly


class AssemblyLoader:
    """
    Loads a DataAssembly from a file.
    """

    def __init__(self, cls, file_path, group=None, **kwargs):
        self.assembly_class = cls
        self.file_path = file_path
        self.group = group

    def load(self):
        result = xr.open_dataarray(self.file_path, group=self.group)
        result = self.correct_stimulus_id_name(result)
        result = self.assembly_class(data=result)
        return result

    @classmethod
    def correct_stimulus_id_name(cls, assembly):
        names = list(get_metadata(assembly, dims=('presentation',), names_only=True))
        if 'image_id' in names and 'stimulus_id' not in names:
            assembly = assembly.assign_coords(
                stimulus_id=('presentation', assembly['image_id']),
            )
        return assembly


class StimulusReferenceAssemblyLoader(AssemblyLoader):
    """
    Loads an assembly and adds a pointer to a stimulus set.
    """

    def __init__(self, cls, file_path, stimulus_set_identifier=None, stimulus_set=None, **kwargs):
        super(StimulusReferenceAssemblyLoader, self).__init__(cls, file_path, **kwargs)
        self.stimulus_set_identifier = stimulus_set_identifier
        self.stimulus_set = stimulus_set

    def load(self):
        result = super(StimulusReferenceAssemblyLoader, self).load()
        result.attrs["stimulus_set_identifier"] = self.stimulus_set_identifier
        result.attrs["stimulus_set"] = self.stimulus_set
        return result


class StimulusMergeAssemblyLoader(StimulusReferenceAssemblyLoader):
    """
    Loads an assembly and merges in metadata from a stimulus set.
    """

    def load(self):
        result = super(StimulusMergeAssemblyLoader, self).load()
        if self.stimulus_set is not None:
            result = self.merge_stimulus_set_meta(result, self.stimulus_set)
        return result

    def merge_stimulus_set_meta(self, assy, stimulus_set):
        dim_name, index_column = "presentation", "stimulus_id"
        assy = assy.reset_index(list(assy.indexes))
        df_of_coords = pd.DataFrame(coords_for_dim(assy, dim_name))
        cols_to_use = stimulus_set.columns.difference(df_of_coords.columns.difference([index_column]))
        merged = df_of_coords.merge(stimulus_set[cols_to_use], on=index_column, how="left")
        for col in stimulus_set.columns:
            assy[col] = (dim_name, merged[col])
        assy = self.assembly_class(data=assy)
        return assy


class GroupAppendAssemblyLoader(StimulusReferenceAssemblyLoader):
    """
    Loads an assembly plus any included metadata assemblies and a pointer to a stimulus set.
    """

    def load(self):
        result = super(GroupAppendAssemblyLoader, self).load()
        nc = netCDF4.Dataset(self.file_path, "r")
        for group in nc.groups:
            try:
                loader_class = MetadataAssembly.get_loader_class()
                loader = loader_class(
                    cls=MetadataAssembly,
                    file_path=self.file_path,
                    stimulus_set_identifier=self.stimulus_set_identifier,
                    stimulus_set=self.stimulus_set,
                    group=group
                )
                meta = loader.load()
                result.attrs[group] = meta
            except Exception as e:
                _logger.warning(
                    f"netCDF file {self.file_path} contains a group that is not loadable as a MetadataAssembly.  ",
                    exc_info=True
                )
        return result



