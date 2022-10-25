import pytest
from pytest import approx

from brainscore_language import score


@pytest.mark.travis_slow
@pytest.mark.parametrize(
    "model_identifier, benchmark_identifier, expected_score",
    [
        ("distilgpt2", "Futrell2018-pearsonr", approx(0.36144805, abs=0.0005)),
        (
            "distilgpt2",
            "Pereira2018_v2022.243sentences-linreg_pearsonr",
            approx(0.72309996, abs=0.0005),
        ),
        (
            "glove-840b",
            "Pereira2018_v2022.384sentences-linreg_pearsonr",
            approx(0.21466593, abs=0.0005),
        ),
        ("gpt2-xl", "Futrell2018-pearsonr", approx(0.31825621, abs=0.0005)),
    ],
)
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(
        model_identifier=model_identifier, benchmark_identifier=benchmark_identifier
    )
    assert actual_score == expected_score


if __name__ == "__main__":
    test_score(
        "distilgpt2",
        "Pereira2018_v2022.243sentences-linreg_pearsonr",
        approx(0.72309996, abs=0.0005),
    )
