from collections import defaultdict

import logging
import numpy as np
import sys
import warnings
from numpy import AxisError
from numpy.random import RandomState
from result_caching import store
from scipy.optimize import curve_fit
from tqdm import tqdm, trange

from brainio.assemblies import DataAssembly, array_is_element, walk_coords, merge_data_arrays
from brainscore_core.metrics import Score
from brainscore_language import load_benchmark
from brainscore_language.benchmark_helpers import ci_error, manual_merge
from brainscore_language.utils import fullname
from brainscore_language.utils.s3 import upload_data_assembly
from brainscore_language.utils.transformations import apply_aggregate

_logger = logging.getLogger(__name__)

"""
Because ceiling estimations for Pereira2018-linear benchmarks are very costly to run (many hours), we pre-compute them,
and store the resulting ceiling (including per-neuroid estimates) on S3.
The code in this file was run only once, and is kept here for reference.
"""


def upload_ceiling(experiment):
    benchmark = load_benchmark(f'Pereira2018.{experiment}-linear')
    ceiler = ExtrapolationCeiling()
    ceiling = ceiler(benchmark.data, metric=benchmark.metric)
    _logger.info(f"Uploading ceiling {ceiling}")
    # Note that because we cannot serialize complex objects to netCDF, attributes like 'raw' and 'bootstrapped_params'
    # will get lost during upload
    upload_data_assembly(ceiling,
                         assembly_identifier=benchmark.identifier,
                         assembly_prefix='ceiling_')
    # also upload raw (and raw.raw) attributes
    upload_data_assembly(ceiling.raw,
                         assembly_identifier=benchmark.identifier,
                         assembly_prefix='ceiling_raw_')
    upload_data_assembly(ceiling.raw.raw,
                         assembly_identifier=benchmark.identifier,
                         assembly_prefix='ceiling_raw_raw_')


def v(x, v0, tau0):  # function for ceiling extrapolation
    return v0 * (1 - np.exp(-x / tau0))


class HoldoutSubjectCeiling:
    def __init__(self, subject_column):
        self.subject_column = subject_column
        self._logger = logging.getLogger(fullname(self))
        self._rng = RandomState(0)
        self._num_bootstraps = 5

    def __call__(self, assembly, metric):
        subjects = set(assembly[self.subject_column].values)
        scores = []
        iterate_subjects = self._rng.choice(list(subjects), size=self._num_bootstraps)  # use only a subset of subjects
        for subject in tqdm(iterate_subjects, desc='heldout subject'):
            try:
                subject_assembly = assembly[{'neuroid': [subject_value == subject
                                                         for subject_value in assembly[self.subject_column].values]}]
                # run subject pool as neural candidate
                subject_pool = subjects - {subject}
                pool_assembly = assembly[
                    {'neuroid': [subject in subject_pool for subject in assembly[self.subject_column].values]}]
                score = metric(pool_assembly, subject_assembly)
                # store scores
                apply_raw = 'raw' in score.attrs and \
                            not hasattr(score.raw, self.subject_column)  # only propagate if column not part of score
                score = score.expand_dims(self.subject_column, _apply_raw=apply_raw)
                score.__setitem__(self.subject_column, [subject], _apply_raw=apply_raw)
                scores.append(score)
            except NoOverlapException as e:
                self._logger.debug(f"Ignoring no overlap {e}")
                continue  # ignore
            except ValueError as e:
                if "Found array with" in str(e):
                    self._logger.debug(f"Ignoring empty array {e}")
                    continue
                else:
                    raise e

        scores = Score.merge(*scores)
        scores = apply_aggregate(lambda scores: scores.mean(self.subject_column), scores)
        return scores


class ExtrapolationCeiling:
    def __init__(self, subject_column='subject', extrapolation_dimension='neuroid',
                 num_bootstraps=100):
        self._logger = logging.getLogger(fullname(self))
        self.subject_column = subject_column
        self.extrapolation_dimension = extrapolation_dimension
        self.num_bootstraps = num_bootstraps
        self.num_subsamples = 10
        self.holdout_ceiling = HoldoutSubjectCeiling(subject_column=subject_column)
        self._rng = RandomState(0)

    def __call__(self, assembly, metric):
        scores = self.collect(identifier=assembly.identifier, assembly=assembly, metric=metric)
        return self.extrapolate(identifier=assembly.identifier, ceilings=scores)

    @store(identifier_ignore=['assembly', 'metric'])
    def collect(self, identifier, assembly, metric):
        self._logger.debug("Collecting data for extrapolation")
        subjects = set(assembly[self.subject_column].values)
        subject_subsamples = tuple(range(2, len(subjects) + 1))
        scores = []
        for num_subjects in tqdm(subject_subsamples, desc='num subjects'):
            for sub_subjects in self._random_combinations(
                    subjects=set(assembly[self.subject_column].values),
                    num_subjects=num_subjects, choice=self.num_subsamples, rng=self._rng):
                sub_assembly = assembly[{'neuroid': [subject in sub_subjects
                                                     for subject in assembly[self.subject_column].values]}]
                selections = {self.subject_column: sub_subjects}
                try:
                    score = self.holdout_ceiling(assembly=sub_assembly, metric=metric)
                    score = score.expand_dims('num_subjects')
                    score['num_subjects'] = [num_subjects]
                    for key, selection in selections.items():
                        expand_dim = f'sub_{key}'
                        score = score.expand_dims(expand_dim)
                        score[expand_dim] = [str(selection)]
                    scores.append(score.raw)
                except KeyError as e:  # nothing to merge
                    if str(e) == "'z'":
                        self._logger.debug(f"Ignoring merge error {e}")
                        continue
                    else:
                        raise e
        scores = Score.merge(*scores)
        return scores

    def _random_combinations(self, subjects, num_subjects, choice, rng):
        # following https://stackoverflow.com/a/55929159/2225200. Also see similar method in `behavioral.py`.
        subjects = np.array(list(subjects))
        combinations = set()
        while len(combinations) < choice:
            elements = rng.choice(subjects, size=num_subjects, replace=False)
            combinations.add(tuple(elements))
        return combinations

    def extrapolate(self, identifier, ceilings):
        neuroid_ceilings = []
        raw_keys = ['bootstrapped_params', 'error_low', 'error_high', 'endpoint_x']
        raw_attrs = defaultdict(list)
        for i in trange(len(ceilings[self.extrapolation_dimension]),
                        desc=f'{self.extrapolation_dimension} extrapolations'):
            try:
                # extrapolate per-neuroid ceiling
                neuroid_ceiling = ceilings.isel(**{self.extrapolation_dimension: [i]})
                extrapolated_ceiling = self.extrapolate_neuroid(neuroid_ceiling.squeeze())
                extrapolated_ceiling = self.add_neuroid_meta(extrapolated_ceiling, neuroid_ceiling)
                neuroid_ceilings.append(extrapolated_ceiling)
                # keep track of raw attributes
                for key in raw_keys:
                    values = extrapolated_ceiling.attrs[key]
                    values = self.add_neuroid_meta(values, neuroid_ceiling)
                    raw_attrs[key].append(values)
            except AxisError:  # no extrapolation successful (happens for 1 neuroid in Pereira)
                _logger.warning(f"Failed to extrapolate neuroid ceiling {i}", exc_info=True)
                continue

        # merge and add meta
        self._logger.debug("Merging neuroid ceilings")
        neuroid_ceilings = manual_merge(*neuroid_ceilings, on=self.extrapolation_dimension)
        neuroid_ceilings.attrs['raw'] = ceilings

        for key, values in raw_attrs.items():
            self._logger.debug(f"Merging {key}")
            values = manual_merge(*values, on=self.extrapolation_dimension)
            neuroid_ceilings.attrs[key] = values
        # aggregate
        ceiling = self.aggregate_neuroid_ceilings(neuroid_ceilings, raw_keys=raw_keys)
        ceiling.attrs['identifier'] = identifier
        return ceiling

    def add_neuroid_meta(self, target, source):
        target = target.expand_dims(self.extrapolation_dimension)
        for coord, dims, values in walk_coords(source):
            if array_is_element(dims, self.extrapolation_dimension):
                target[coord] = dims, values
        return target

    def aggregate_neuroid_ceilings(self, neuroid_ceilings, raw_keys):
        ceiling = neuroid_ceilings.median(self.extrapolation_dimension)
        ceiling.attrs['raw'] = neuroid_ceilings
        for key in raw_keys:
            values = neuroid_ceilings.attrs[key]
            aggregate = values.median(self.extrapolation_dimension)
            if not aggregate.shape:  # scalar value, e.g. for error_low
                aggregate = aggregate.item()
            ceiling.attrs[key] = aggregate
        return ceiling

    def extrapolate_neuroid(self, ceilings):
        # figure out how many extrapolation x points we have. E.g. for Pereira, not all combinations are possible
        subject_subsamples = list(sorted(set(ceilings['num_subjects'].values)))
        rng = RandomState(0)
        bootstrap_params = []
        for bootstrap in range(self.num_bootstraps):
            bootstrapped_scores = []
            for num_subjects in subject_subsamples:
                num_scores = ceilings.sel(num_subjects=num_subjects)
                # the sub_subjects dimension creates nans, get rid of those
                num_scores = num_scores.dropna(f'sub_{self.subject_column}')
                assert set(num_scores.dims) == {f'sub_{self.subject_column}', 'split'} or \
                       set(num_scores.dims) == {f'sub_{self.subject_column}'}
                # choose from subject subsets and the splits therein, with replacement for variance
                choices = num_scores.values.flatten()
                bootstrapped_score = rng.choice(choices, size=len(choices), replace=True)
                bootstrapped_scores.append(np.mean(bootstrapped_score))

            try:
                params = self.fit(subject_subsamples, bootstrapped_scores)
            except RuntimeError:  # optimal parameters not found
                params = [np.nan, np.nan]
            params = DataAssembly([params], coords={'bootstrap': [bootstrap], 'param': ['v0', 'tau0']},
                                  dims=['bootstrap', 'param'])
            bootstrap_params.append(params)
        bootstrap_params = merge_data_arrays(bootstrap_params)
        # find endpoint and error
        asymptote_threshold = .0005
        interpolation_xs = np.arange(1000)
        ys = np.array([v(interpolation_xs, *params) for params in bootstrap_params.values
                       if not np.isnan(params).any()])
        median_ys = np.median(ys, axis=0)
        diffs = np.diff(median_ys)
        end_x = np.where(diffs < asymptote_threshold)[0].min()  # first x where increase smaller than threshold
        # put together
        center = np.median(np.array(bootstrap_params)[:, 0])
        error_low, error_high = ci_error(ys[:, end_x], center=center)
        score = Score(center)
        score.attrs['raw'] = ceilings
        score.attrs['error_low'] = DataAssembly(error_low)
        score.attrs['error_high'] = DataAssembly(error_high)
        score.attrs['bootstrapped_params'] = bootstrap_params
        score.attrs['endpoint_x'] = DataAssembly(end_x)
        return score

    def fit(self, subject_subsamples, bootstrapped_scores):
        valid = ~np.isnan(bootstrapped_scores)
        if sum(valid) < 1:
            raise RuntimeError("No valid scores in sample")
        params, pcov = curve_fit(v, subject_subsamples, bootstrapped_scores,
                                 # v (i.e. max ceiling) is between 0 and 1, tau0 unconstrained
                                 bounds=([0, -np.inf], [1, np.inf]))
        return params


class NoOverlapException(Exception):
    pass


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    for shush_logger in ['botocore', 'boto3', 's3transfer', 'urllib3']:
        logging.getLogger(shush_logger).setLevel(logging.INFO)
    warnings.filterwarnings("ignore")  # suppress RuntimeWarning from extrapolation overflows

    upload_ceiling('243sentences')
    upload_ceiling('384sentences')
