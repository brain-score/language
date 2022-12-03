import pytest

from brainscore_language.model_helpers import preprocessing


@pytest.mark.parametrize(
    "context, expected",
    [
        (["the quick", ", and sneaky", "", "brown fox"],
          "the quick, and sneaky brown fox"),
    ]
)
def test_prepare_context(context, expected):
    assert preprocessing.prepare_context(context) == expected