import numpy as np

# moved from brain-score.transformations to also use the subset algorithm for new data sets in brainio-contrib
from brainio.assemblies import walk_coords


def subset(source_assembly, target_assembly, subset_dims=None, dims_must_match=True, repeat=False):
    """
    Returns the subset of the source_assembly whose coordinates align with those specified by target_assembly.
    Ordering is not guaranteed.
    :param subset_dims: either dimensions, then all its levels will be used or levels right away
    :param dims_must_match:
    :return:
    """
    subset_dims = subset_dims or target_assembly.dims
    for dim in subset_dims:
        assert hasattr(target_assembly, dim)
        assert hasattr(source_assembly, dim)
        # we assume here that it does not matter if all levels are present in the source assembly
        # as long as there is at least one level that we can select over
        levels = target_assembly[dim].variable.level_names or [dim]
        assert any(hasattr(source_assembly, level) for level in levels)
        for level in levels:
            if not hasattr(source_assembly, level):
                continue
            target_values = target_assembly[level].values
            source_values = source_assembly[level].values
            if repeat:
                indexer = index_efficient(source_values, target_values)
            else:
                indexer = np.array([val in target_values for val in source_values])
                indexer = np.where(indexer)[0]
            if dim not in target_assembly.dims:
                # not actually a dimension, but rather a coord -> filter along underlying dim
                dim = target_assembly[dim].dims
                assert len(dim) == 1
                dim = dim[0]
            dim_indexes = {_dim: slice(None) if _dim != dim else indexer for _dim in source_assembly.dims}
            if len(np.unique(source_assembly.dims)) == len(source_assembly.dims):  # no repeated dimensions
                source_assembly = source_assembly.isel(**dim_indexes)
                continue
            # work-around when dimensions are repeated. `isel` will keep only the first instance of a repeated dimension
            positional_dim_indexes = [dim_indexes[dim] for dim in source_assembly.dims]
            coords = {}
            for coord, dims, value in walk_coords(source_assembly):
                if len(dims) == 1:
                    coords[coord] = (dims, value[dim_indexes[dims[0]]])
                elif len(dims) == 0:
                    coords[coord] = (dims, value)
                elif len(set(dims)) == 1:
                    coords[coord] = (dims, value[np.ix_(*[dim_indexes[dim] for dim in dims])])
                else:
                    raise NotImplementedError("cannot handle multiple dimensions")
            source_assembly = type(source_assembly)(source_assembly.values[np.ix_(*positional_dim_indexes)],
                                                    coords=coords, dims=source_assembly.dims)
        if dims_must_match:
            # dims match up after selection. cannot compare exact equality due to potentially missing levels
            assert len(target_assembly[dim]) == len(source_assembly[dim])
    return source_assembly


def index_efficient(source_values, target_values):
    source_sort_indices, target_sort_indices = np.argsort(source_values), np.argsort(target_values)
    source_values, target_values = source_values[source_sort_indices], target_values[target_sort_indices]
    indexer = []
    source_index, target_index = 0, 0
    while target_index < len(target_values) and source_index < len(source_values):
        if source_values[source_index] == target_values[target_index]:
            indexer.append(source_sort_indices[source_index])
            # if next source value is greater than target, use next target. else next source.
            # if target values remain the same, we might as well take the next target.
            if (target_index + 1 < len(target_values) and
                target_values[target_index + 1] == target_values[target_index]) or \
                    (source_index + 1 < len(source_values) and
                     source_values[source_index + 1] > target_values[target_index]):
                target_index += 1
            else:
                source_index += 1
        elif source_values[source_index] < target_values[target_index]:
            source_index += 1
        else:  # source_values[source_index] > target_values[target_index]:
            target_index += 1
    return indexer
