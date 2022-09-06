import pytest
from pytest import approx

from brainscore_language import load_model, load_benchmark


@pytest.mark.travis_slow
@pytest.mark.parametrize('model_identifier, benchmark_identifier, expected_score', [
    ('distilgpt2', 'Futrell2018-pearsonr', approx(0.36144805, abs=.0005)),
    ('gpt2-xl', 'Futrell2018-pearsonr', approx(0.31825621, abs=.0005)),
])
def test_model_benchmark(model_identifier, benchmark_identifier, expected_score):
    model = load_model(model_identifier)
    benchmark = load_benchmark(benchmark_identifier)
    score = benchmark(model)
    assert score == expected_score
