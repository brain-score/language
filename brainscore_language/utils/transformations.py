import logging
import math
import numpy as np
import xarray as xr
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, KFold, StratifiedKFold
from tqdm import tqdm

from brainio.assemblies import walk_coords
from brainio.transform import subset
from brainscore_core.metrics import Score
from . import fullname


class Transformation:
    """
    Transforms an incoming assembly into parts/combinations thereof,
    yields them for further processing,
    and packages the results back together.
    """

    def __call__(self, *args, apply, aggregate=None, **kwargs) -> Score:
        values = self._run_pipe(*args, apply=apply, **kwargs)

        score = apply_aggregate(aggregate, values) if aggregate is not None else values
        score = apply_aggregate(self.aggregate, score)
        return score

    def _run_pipe(self, *args, apply, **kwargs):
        generator = self.pipe(*args, **kwargs)
        for vals in generator:
            y = apply(*vals)
            done = generator.send(y)
            if done:
                break
        result = next(generator)
        return result

    def pipe(self, *args, **kwargs):
        raise NotImplementedError()

    def _get_result(self, *args, done):
        """
        Yields the `*args` for further processing by coroutines
        and waits for the result to be sent back.
        :param args: transformed values
        :param bool done: whether this is the last transformation and the next `yield` is the combined result
        :return: the result from processing by the coroutine
        """
        result = yield args  # yield the values to coroutine
        yield done  # wait for coroutine to send back similarity and inform whether result is ready to be returned
        return result

    def aggregate(self, score: Score) -> Score:
        return Score(score)


class Split:
    class Defaults:
        splits = 10
        train_size = .9
        split_coord = 'stimulus_id'
        stratification_coord = None
        unique_split_values = False
        random_state = 1

    def __init__(self,
                 splits=Defaults.splits, train_size=None, test_size=None,
                 split_coord=Defaults.split_coord, stratification_coord=Defaults.stratification_coord, kfold=False,
                 unique_split_values=Defaults.unique_split_values, random_state=Defaults.random_state):
        super().__init__()
        if train_size is None and test_size is None:
            train_size = self.Defaults.train_size
        if kfold:
            assert (train_size is None or train_size == self.Defaults.train_size) and test_size is None
            if stratification_coord:
                self._split = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
            else:
                self._split = KFold(n_splits=splits, shuffle=True, random_state=random_state)
        else:
            if stratification_coord:
                self._split = StratifiedShuffleSplit(
                    n_splits=splits, train_size=train_size, test_size=test_size, random_state=random_state)
            else:
                self._split = ShuffleSplit(
                    n_splits=splits, train_size=train_size, test_size=test_size, random_state=random_state)
        self._split_coord = split_coord
        self._stratification_coord = stratification_coord
        self._unique_split_values = unique_split_values

        self._logger = logging.getLogger(fullname(self))

    @property
    def do_stratify(self):
        return bool(self._stratification_coord)

    def build_splits(self, assembly):
        cross_validation_values, indices = extract_coord(assembly, self._split_coord, unique=self._unique_split_values)
        data_shape = np.zeros(len(cross_validation_values))
        args = [assembly[self._stratification_coord].values[indices]] if self.do_stratify else []
        splits = self._split.split(data_shape, *args)
        return cross_validation_values, list(splits)

    @classmethod
    def aggregate(cls, values):
        center = values.mean('split')
        error = standard_error_of_the_mean(values, 'split')
        return Score([center, error],
                     coords={**{'aggregation': ['center', 'error']},
                             **{coord: (dims, values) for coord, dims, values in walk_coords(center)}},
                     dims=('aggregation',) + center.dims)


def extract_coord(assembly, coord, unique=False):
    if not unique:
        coord_values = assembly[coord].values
        indices = list(range(len(coord_values)))
    else:
        # need unique values for when e.g. repetitions are heavily redundant and splits would yield equal unique values
        coord_values, indices = np.unique(assembly[coord].values, return_index=True)
    dims = assembly[coord].dims
    assert len(dims) == 1
    extracted_assembly = xr.DataArray(coord_values, coords={coord: coord_values}, dims=[coord])
    extracted_assembly = extracted_assembly.stack(**{dims[0]: (coord,)})
    return extracted_assembly if not unique else extracted_assembly, indices


class TestOnlyCrossValidationSingle:
    def __init__(self, *args, **kwargs):
        self._cross_validation = CrossValidationSingle(*args, **kwargs)

    def __call__(self, *args, apply, **kwargs):
        apply_wrapper = lambda train, test: apply(test)
        return self._cross_validation(*args, apply=apply_wrapper, **kwargs)


class TestOnlyCrossValidation:
    def __init__(self, *args, **kwargs):
        self._cross_validation = CrossValidation(*args, **kwargs)

    def __call__(self, *args, apply, **kwargs):
        apply_wrapper = lambda train1, train2, test1, test2: apply(test1, test2)
        return self._cross_validation(*args, apply=apply_wrapper, **kwargs)


class CrossValidationSingle(Transformation):
    def __init__(self,
                 splits=Split.Defaults.splits, train_size=None, test_size=None,
                 split_coord=Split.Defaults.split_coord, stratification_coord=Split.Defaults.stratification_coord,
                 unique_split_values=Split.Defaults.unique_split_values, random_state=Split.Defaults.random_state):
        super().__init__()
        self._split = Split(splits=splits, split_coord=split_coord,
                            stratification_coord=stratification_coord, unique_split_values=unique_split_values,
                            train_size=train_size, test_size=test_size, random_state=random_state)
        self._logger = logging.getLogger(fullname(self))

    def pipe(self, assembly):
        """
        :param assembly: the assembly to cross-validate over
        """
        cross_validation_values, splits = self._split.build_splits(assembly)

        split_scores = []
        for split_iterator, (train_indices, test_indices), done \
                in tqdm(enumerate_done(splits), total=len(splits), desc='cross-validation'):
            train_values, test_values = cross_validation_values[train_indices], cross_validation_values[test_indices]
            train = subset(assembly, train_values, dims_must_match=False)
            test = subset(assembly, test_values, dims_must_match=False)

            split_score = yield from self._get_result(train, test, done=done)
            split_score = split_score.expand_dims('split')
            split_score['split'] = [split_iterator]
            split_scores.append(split_score)

        split_scores = Score.merge(*split_scores)
        yield split_scores

    def aggregate(self, score):
        return self._split.aggregate(score)


class CrossValidation(Transformation):
    """
    Performs multiple splits over a source and target assembly.
    No guarantees are given for data-alignment, use the metadata.
    """

    def __init__(self, *args, split_coord=Split.Defaults.split_coord,
                 stratification_coord=Split.Defaults.stratification_coord, **kwargs):
        self._split_coord = split_coord
        self._stratification_coord = stratification_coord
        self._split = Split(*args, split_coord=split_coord, stratification_coord=stratification_coord, **kwargs)
        self._logger = logging.getLogger(fullname(self))

    def pipe(self, source_assembly, target_assembly):
        # check only for equal values, alignment is given by metadata
        assert sorted(source_assembly[self._split_coord].values) == sorted(target_assembly[self._split_coord].values)
        if self._split.do_stratify:
            assert hasattr(source_assembly, self._stratification_coord)
            assert sorted(source_assembly[self._stratification_coord].values) == \
                   sorted(target_assembly[self._stratification_coord].values)
        cross_validation_values, splits = self._split.build_splits(target_assembly)

        split_scores = []
        for split_iterator, (train_indices, test_indices), done \
                in tqdm(enumerate_done(splits), total=len(splits), desc='cross-validation'):
            train_values, test_values = cross_validation_values[train_indices], cross_validation_values[test_indices]
            train_source = subset(source_assembly, train_values, dims_must_match=False)
            train_target = subset(target_assembly, train_values, dims_must_match=False)
            assert len(train_source[self._split_coord]) == len(train_target[self._split_coord])
            test_source = subset(source_assembly, test_values, dims_must_match=False)
            test_target = subset(target_assembly, test_values, dims_must_match=False)
            assert len(test_source[self._split_coord]) == len(test_target[self._split_coord])

            split_score = yield from self._get_result(train_source, train_target, test_source, test_target,
                                                      done=done)
            split_score = split_score.expand_dims('split')
            split_score['split'] = [split_iterator]
            split_scores.append(split_score)

        split_scores = Score.merge(*split_scores)
        yield split_scores

    def aggregate(self, score):
        return self._split.aggregate(score)


def apply_aggregate(aggregate_fnc, values: Score) -> Score:
    """
    Applies the aggregate while keeping the raw values in the attrs.
    If raw values are already present, keeps them, else they are added.
    """
    score = aggregate_fnc(values)
    # make sure we maintain all the raw attributes in the score object
    for attr_key in values[0].attrs:
        if attr_key not in score.attrs:
            score.attrs[attr_key] = values.attrs[attr_key]
    # if there is not already a raw attribute on the aggregated score, keep the input to this function.
    if Score.RAW_VALUES_KEY not in score.attrs:
        score.attrs[Score.RAW_VALUES_KEY] = values
    return score


def standard_error_of_the_mean(values, dim):
    return values.std(dim) / math.sqrt(len(values[dim]))


def enumerate_done(values):
    for i, val in enumerate(values):
        done = i == len(values) - 1
        yield i, val, done
