import pytest
from pytest import approx

from brainscore_language import load_model, load_benchmark


@pytest.mark.travis_slow
@pytest.mark.parametrize('model_identifier, benchmark_identifier, expected_score', [
    ('distilgpt2', 'Futrell2018-pearsonr', approx(.01988352, abs=.005)),
])
def test_model_benchmark(model_identifier, benchmark_identifier, expected_score):
    model = load_model(model_identifier)
    benchmark = load_benchmark(benchmark_identifier)
    score = benchmark(model)
    assert score == expected_score
