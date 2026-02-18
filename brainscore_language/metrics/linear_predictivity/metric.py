import numpy as np
import scipy.linalg
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import scale

from brainscore_core.supported_data_standards.brainio.assemblies import NeuroidAssembly, array_is_element, DataAssembly
from brainscore_core.supported_data_standards.brainio.assemblies import walk_coords
from brainscore_core.metrics import Score, Metric
from brainscore_language.utils.transformations import CrossValidation
from brainscore_language.metrics.linear_predictivity.ridgecv_gpu import RidgeGCVTorch


class Defaults:
    expected_dims = ('presentation', 'neuroid')
    stimulus_coord = 'stimulus_id'
    neuroid_dim = 'neuroid'
    neuroid_coord = 'neuroid_id'


class XarrayRegression:
    """
    Adds alignment-checking, un- and re-packaging, and comparison functionality to a regression.
    """

    def __init__(self, regression, expected_dims=Defaults.expected_dims, neuroid_dim=Defaults.neuroid_dim,
                 neuroid_coord=Defaults.neuroid_coord, stimulus_coord=Defaults.stimulus_coord):
        self._regression = regression
        self._expected_dims = expected_dims
        self._neuroid_dim = neuroid_dim
        self._neuroid_coord = neuroid_coord
        self._stimulus_coord = stimulus_coord
        self._target_neuroid_values = None

    def fit(self, source, target):
        source, target = self._align(source), self._align(target)
        source, target = source.sortby(self._stimulus_coord), target.sortby(self._stimulus_coord)

        self._regression.fit(source, target)

        self._target_neuroid_values = {}
        for name, dims, values in walk_coords(target):
            if self._neuroid_dim in dims:
                assert array_is_element(dims, self._neuroid_dim)
                self._target_neuroid_values[name] = values

    def predict(self, source):
        source = self._align(source)
        predicted_values = self._regression.predict(source)
        prediction = self._package_prediction(predicted_values, source=source)
        return prediction

    def _package_prediction(self, predicted_values, source):
        coords = {coord: (dims, values) for coord, dims, values in walk_coords(source)
                  if not array_is_element(dims, self._neuroid_dim)}
        # re-package neuroid coords
        dims = source.dims
        # if there is only one neuroid coordinate, it would get discarded and the dimension would be used as coordinate.
        # to avoid this, we can build the assembly first and then stack on the neuroid dimension.
        neuroid_level_dim = None
        if len(self._target_neuroid_values) == 1:  # extract single key: https://stackoverflow.com/a/20145927/2225200
            (neuroid_level_dim, _), = self._target_neuroid_values.items()
            dims = [dim if dim != self._neuroid_dim else neuroid_level_dim for dim in dims]
        for target_coord, target_value in self._target_neuroid_values.items():
            # this might overwrite values which is okay
            coords[target_coord] = (neuroid_level_dim or self._neuroid_dim), target_value
        prediction = NeuroidAssembly(predicted_values, coords=coords, dims=dims)
        if neuroid_level_dim:
            prediction = prediction.stack(**{self._neuroid_dim: [neuroid_level_dim]})

        return prediction

    def _align(self, assembly):
        assert set(assembly.dims) == set(self._expected_dims), \
            f"Expected {set(self._expected_dims)}, but got {set(assembly.dims)}"
        return assembly.transpose(*self._expected_dims)


class XarrayCorrelation:
    def __init__(self, correlation, correlation_coord=Defaults.stimulus_coord, neuroid_coord=Defaults.neuroid_coord):
        self._correlation = correlation
        self._correlation_coord = correlation_coord
        self._neuroid_coord = neuroid_coord

    def __call__(self, prediction, target) -> Score:
        # align
        prediction = prediction.sortby([self._correlation_coord, self._neuroid_coord])
        target = target.sortby([self._correlation_coord, self._neuroid_coord])
        assert np.array(prediction[self._correlation_coord].values == target[self._correlation_coord].values).all()
        assert np.array(prediction[self._neuroid_coord].values == target[self._neuroid_coord].values).all()
        # compute correlation per neuroid
        neuroid_dims = target[self._neuroid_coord].dims
        assert len(neuroid_dims) == 1
        correlations = []
        for i, coord_value in enumerate(target[self._neuroid_coord].values):
            target_neuroids = target.isel(**{neuroid_dims[0]: i})  # `isel` is about 10x faster than `sel`
            prediction_neuroids = prediction.isel(**{neuroid_dims[0]: i})
            r, p = self._correlation(target_neuroids, prediction_neuroids)
            correlations.append(r)
        # package
        result = Score(correlations,
                       coords={coord: (dims, values)
                               for coord, dims, values in walk_coords(target) if dims == neuroid_dims},
                       dims=neuroid_dims)
        return result

def pearsonr(x, y):
    xmean = x.mean(axis=0, keepdims=True)
    ymean = y.mean(axis=0, keepdims=True)

    xm = x - xmean
    ym = y - ymean

    normxm = scipy.linalg.norm(xm, axis=0, keepdims=True) + 1e-8
    normym = scipy.linalg.norm(ym, axis=0, keepdims=True) + 1e-8

    r = ((xm / normxm) * (ym / normym)).sum(axis=0)

    return r

class XarrayCorrelationBatched:
    def __init__(self, correlation_coord=Defaults.stimulus_coord, neuroid_coord=Defaults.neuroid_coord):
        self._correlation = pearsonr
        self._correlation_coord = correlation_coord
        self._neuroid_coord = neuroid_coord

    def __call__(self, prediction, target):
        # align
        prediction = prediction.sortby([self._correlation_coord, self._neuroid_coord])
        target = target.sortby([self._correlation_coord, self._neuroid_coord])
        assert np.array(prediction[self._correlation_coord].values == target[self._correlation_coord].values).all()
        assert np.array(prediction[self._neuroid_coord].values == target[self._neuroid_coord].values).all()
        # compute correlation per neuroid
        neuroid_dims = target[self._neuroid_coord].dims
        assert len(neuroid_dims) == 1
        prediction = prediction.transpose(..., *neuroid_dims)
        target = target.transpose(..., *neuroid_dims)
        correlations = self._correlation(prediction.values, target.values)
        # package
        result = Score(correlations,
                       coords={coord: (dims, values)
                               for coord, dims, values in walk_coords(target) if dims == neuroid_dims},
                       dims=neuroid_dims)
        return result

class CrossRegressedCorrelation(Metric):
    def __init__(self, regression, correlation, crossvalidation_kwargs=None, store_regression_weights=False):
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}

        self.cross_validation = CrossValidation(**crossvalidation_kwargs)
        self.regression = regression
        self.correlation = correlation
        self.store_regression_weights = store_regression_weights

    def __call__(self, assembly1: DataAssembly, assembly2: DataAssembly) -> Score:
        return self.cross_validation(assembly1, assembly2, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source_train, target_train, source_test, target_test):
        self.regression.fit(source_train, target_train)
        prediction = self.regression.predict(source_test)
        score = self.correlation(prediction, target_test)
        if self.store_regression_weights:
            self.attach_regression_weights(score=score, source_test=source_test, target_test=target_test)
        return score

    def attach_regression_weights(self, score, source_test, target_test):
        source_weight_dim = source_test.dims[-1]
        target_weight_dim = target_test.dims[-1]
        coef = DataAssembly(self.regression._regression.coef_,
                            coords={
                                **{f'source_{coord}': (f'source_{source_weight_dim}', values)
                                   for coord, dims, values in walk_coords(source_test[source_weight_dim])},
                                **{f'target_{coord}': (f'target_{target_weight_dim}', values)
                                   for coord, dims, values in walk_coords(target_test[target_weight_dim])},
                            },
                            dims=[f'source_{source_weight_dim}', f'target_{target_weight_dim}'])
        score.attrs['raw_regression_coef'] = coef
        intercept = DataAssembly(self.regression._regression.intercept_,
                                 coords={f'target_{coord}': (f'target_{target_weight_dim}', values)
                                         for coord, dims, values in walk_coords(target_test[target_weight_dim])},
                                 dims=[f'target_{target_weight_dim}'])
        score.attrs['raw_regression_intercept'] = intercept

    def aggregate(self, scores):
        return scores.median(dim='neuroid')


class ScaledCrossRegressedCorrelation(Metric):
    def __init__(self, *args, **kwargs):
        self.cross_regressed_correlation = CrossRegressedCorrelation(*args, **kwargs)
        self.aggregate = self.cross_regressed_correlation.aggregate

    def __call__(self, source: DataAssembly, target: DataAssembly) -> Score:
        scaled_values = scale(target, copy=True)
        target = target.__class__(scaled_values, coords={
            coord: (dims, value) for coord, dims, value in walk_coords(target)}, dims=target.dims)
        return self.cross_regressed_correlation(source, target)

def ridge_regression(xarray_kwargs=None):
    regression = RidgeGCVTorch(alphas=np.logspace(-3, 3, 7))
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression

def linear_regression(xarray_kwargs=None):
    regression = LinearRegression()
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression

def pearsonr_correlation(xarray_kwargs=None):
    # Uses XarrayCorrelationBatched which computes Pearson r via vectorized matrix operations
    # with a +1e-8 epsilon for numerical stability (see `pearsonr` function above).
    # This replaces the previous per-neuroid scipy.stats.pearsonr loop (XarrayCorrelation),
    # producing slightly different values due to the epsilon normalization.
    xarray_kwargs = xarray_kwargs or {}
    return XarrayCorrelationBatched(**xarray_kwargs)

def linear_pearsonr(*args, regression_kwargs=None, correlation_kwargs=None, **kwargs):
    regression = linear_regression(regression_kwargs or {})
    correlation = pearsonr_correlation(correlation_kwargs or {})
    return CrossRegressedCorrelation(*args, regression=regression, correlation=correlation, **kwargs)

def ridge_pearsonr(*args, regression_kwargs=None, correlation_kwargs=None, **kwargs):
    regression = ridge_regression(regression_kwargs or {})
    correlation = pearsonr_correlation(correlation_kwargs or {})
    return CrossRegressedCorrelation(*args, regression=regression, correlation=correlation, **kwargs)