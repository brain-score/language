import pytest
import subprocess
import sys
from pathlib import Path
from pytest import approx

from brainscore_language import score


@pytest.mark.travis_slow
@pytest.mark.parametrize(
    "model_identifier, benchmark_identifier, expected_score",
    [
        ("distilgpt2", "Futrell2018-pearsonr", approx(0.36144805, abs=0.0005)),
        ("distilgpt2", "Pereira2018.243sentences-linear", approx(0.72309996, abs=0.0005)),
        ("glove-840b", "Pereira2018.384sentences-linear", approx(0.18385368, abs=0.0005)),
        ("gpt2-xl", "Futrell2018-pearsonr", approx(0.31825621, abs=0.0005)),
    ],
)
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier,
                         install_dependencies="newenv")
    assert actual_score == expected_score


def test_commandline_score():
    process = subprocess.run(
        [
            sys.executable,
            "brainscore_language",
            "score",
            "--model_identifier=randomembedding-100",
            "--benchmark_identifier=Pereira2018.243sentences-linear",
        ],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, "Process failed"
    assert "error" not in process.stderr.lower()
    output = process.stdout
    assert "Score" in output
    assert "0.0285" in output
    assert "<xarray.Score ()>\narray(0.0285022)" in output
    assert "model_identifier:      randomembedding-100" in output
    assert "benchmark_identifier:  Pereira2018.243sentences-linear" in output
