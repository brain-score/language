import pytest
from pytest import approx

from brainscore_language import score


@pytest.mark.travis_slow
@pytest.mark.parametrize('model_identifier, benchmark_identifier, expected_score', [
    ('distilgpt2', 'Futrell2018-pearsonr', approx(0.36144805, abs=.0005)),
    ('distilgpt2', 'Pereira2018.243sentences-linear', approx(0.72309996, abs=.0005)),
    ('glove-840b', 'Pereira2018.384sentences-linear', approx(0.18385368, abs=.0005)),
    ('gpt2-xl', 'Futrell2018-pearsonr', approx(0.31825621, abs=.0005)),
])
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier)
    assert actual_score == expected_score
