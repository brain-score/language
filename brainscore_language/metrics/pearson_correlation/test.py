import numpy as np
from pytest import approx

from brainscore_language import load_metric


class TestMetric:
    def test_identical(self):
        a1 = a2 = [1, 2, 3, 4, 5]
        metric = load_metric('pearsonr')
        score = metric(a1, a2)
        assert score == 1

    def test_negative_correlation_is_1(self):
        a1 = np.array([1, 2, 3, 4, 5])
        a2 = -a1
        metric = load_metric('pearsonr')
        score = metric(a1, a2)
        assert score == 1

    def test_weak_correlation(self):
        a1 = [1, 2, 3, 4, 5]
        a2 = [3, 1, 6, 1, 2]
        metric = load_metric('pearsonr')
        score = metric(a1, a2)
        assert score == approx(.152, abs=.005)
