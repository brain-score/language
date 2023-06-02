import numpy as np
import pandas as pd
import pytest
from pprint import pprint
from pytest import approx
from typing import List, Dict

from brainscore_language import load_model
from brainscore_language.benchmarks.syntaxgym import SyntaxGym2020
from brainscore_language.benchmarks.syntaxgym.benchmark import SyntaxGymTSE, SyntaxGymSingleTSE
from brainscore_language.benchmarks.syntaxgym.gpt2_precomputed import REFERENCE_DISTILGPT2_SCORES, \
    REFERENCE_DISTILGPT2_REGION_TOTALS
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject


@pytest.fixture(scope="session")
def distilgpt2():
    return HuggingfaceSubject(model_id="distilgpt2", region_layer_mapping={})


def test_score_aggregate(distilgpt2):
    """
    Scores computed with individual benchmark should match exactly those
    in the combined benchmark class, and be aggregated correctly by mean.
    """
    suites = ["cleft", "cleft_modifier"]
    benchmark = SyntaxGymTSE({suite: suite for suite in suites})

    score = benchmark(distilgpt2)
    sub_scores = {benchmark.suite.meta["name"]: benchmark(distilgpt2)
                  for benchmark in benchmark.sub_benchmarks}

    np.testing.assert_equal(score.item(), np.mean(list(sub_scores.values())))

    assert score.sub_scores["sub_benchmark"].values.tolist() == suites
    np.testing.assert_array_equal(score.sub_scores.values,
                                  [sub_scores[s] for s in suites])


@pytest.mark.parametrize("suite", REFERENCE_DISTILGPT2_REGION_TOTALS.keys())
def test_region_totals_match(distilgpt2, suite: str):
    """
    The region-level surprisals computed on the subordination_src-src
    test suite should match those of the reference implementation.
    """

    benchmark = SyntaxGymSingleTSE(identifier=suite, suite_ref=suite)
    subject = distilgpt2

    expected: List[Dict[str, float]] = REFERENCE_DISTILGPT2_REGION_TOTALS[suite][:6]
    # Reference may be a subset of actual items. Truncate as necessary.
    benchmark.suite.items = benchmark.suite.items[:len(expected)]

    actual = benchmark.get_region_totals(subject)

    keys = actual[0].keys()
    assert set(keys) == set(expected[0].keys())

    # Convert to dataframe for easy comparison + easy visualization of mismatches
    def make_item_df(region_totals):
        return pd.Series(region_totals).unstack() \
            .rename_axis(index="condition", columns="region_number")

    actual_df = pd.concat([make_item_df(item_totals) for item_totals in actual],
                          names=["item_number"],
                          keys=np.arange(len(actual)) + 1).astype(float)
    expected_df = pd.concat([make_item_df(item_totals) for item_totals in expected],
                            names=["item_number"],
                            keys=np.arange(len(actual)) + 1).astype(float)
    pprint((expected_df - actual_df).round(3))
    pd.testing.assert_frame_equal(actual_df, expected_df, atol=1e-3, check_exact=False)


class TestSyntaxGym2020Root:
    def test_number_sub_benchmarks(self):
        assert len(SyntaxGym2020().sub_benchmarks) == 31

    @pytest.mark.travis_slow
    def test_model_score(self):
        model = load_model('distilgpt2')
        benchmark = SyntaxGym2020()
        actual_score = benchmark(model)
        assert actual_score == approx(0.51398774, abs=.0005)


@pytest.mark.parametrize("suite_ref", REFERENCE_DISTILGPT2_SCORES.keys())
def test_suite_accuracies(distilgpt2: HuggingfaceSubject, suite_ref: str):
    """
    Compare distilgpt2 suite accuracies with those of the reference implementation.
    """
    tse = SyntaxGymSingleTSE(identifier=suite_ref, suite_ref=suite_ref)
    accuracy = tse(distilgpt2)
    expected_accuracy = REFERENCE_DISTILGPT2_SCORES[suite_ref]
    np.testing.assert_almost_equal(float(accuracy), expected_accuracy, decimal=3)
