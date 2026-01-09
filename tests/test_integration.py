import os
import pytest
import subprocess
import sys
import torch
from pathlib import Path
from pytest import approx

from brainscore_language import score

# Use more lenient tolerance for MPS (Apple Silicon) due to hardware differences
_SCORE_ATOL = 0.005 


@pytest.mark.travis_slow
@pytest.mark.parametrize(
    "model_identifier, benchmark_identifier, expected_score",
    [
        ("distilgpt2", "Futrell2018-pearsonr", approx(0.36144805, abs=_SCORE_ATOL)),
        ("distilgpt2", "Pereira2018.243sentences-linear", approx(0.82228507, abs=_SCORE_ATOL)),
        ("glove-840b", "Pereira2018.384sentences-linear", approx(0.18385368, abs=_SCORE_ATOL)),
        ("gpt2-xl", "Futrell2018-pearsonr", approx(0.31825621, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-center_embed', approx(0.96428571, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-center_embed_mod', approx(0.92857143, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-cleft', approx(1.0, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-cleft_modifier', approx(0.725, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-fgd_hierarchy', approx(0.0, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-fgd_object', approx(0.875, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-fgd_pp', approx(0.875, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-fgd_subject', approx(0.54166667, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-mvrr', approx(0.821429, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-mvrr_mod', approx(0.785714, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-npi_orc_any', approx(0.026316, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-npi_orc_ever', approx(0.026316, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-npi_src_any', approx(0.0, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-npi_src_ever', approx(0.0, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-npz_ambig', approx(0.66666667, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-npz_ambig_mod', approx(0.66666667, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-npz_obj', approx(0.83333333, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-npz_obj_mod', approx(0.875, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-number_orc', approx(0.10526316, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-number_prep', approx(0.57894737, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-number_src', approx(0.78947368, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-reflexive_orc_fem', approx(0.0, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-reflexive_orc_masc', approx(0.368421, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-reflexive_prep_fem', approx(0.105263, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-reflexive_prep_masc', approx(0.473684, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-reflexive_src_fem', approx(0.157895, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-reflexive_src_masc', approx(0.526316, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-subordination', approx(0.2173913, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-subordination_orc-orc', approx(0.95652174, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-subordination_pp-pp', approx(0.47826087, abs=_SCORE_ATOL)),
        ('distilgpt2', 'syntaxgym-subordination_src-src', approx(0.56521739, abs=_SCORE_ATOL))
    ]
)
def test_score(model_identifier, benchmark_identifier, expected_score):
    actual_score = score(model_identifier=model_identifier, benchmark_identifier=benchmark_identifier)
    assert actual_score == expected_score


@pytest.mark.parametrize(
    "model_identifier, benchmark_identifier, expected_score, install_dependencies",
    [
        ("randomembedding-100", "Pereira2018.243sentences-linear",
         approx(0.0285022, abs=_SCORE_ATOL), "newenv"),
    ]
)
def test_score_with_install_dependencies(
        model_identifier, benchmark_identifier, expected_score, install_dependencies):
    install_dependence_preference = os.environ.get(
        "BS_INSTALL_DEPENDENCIES", "yes")
    os.environ["BS_INSTALL_DEPENDENCIES"] = install_dependencies
    actual_score = score(
        model_identifier=model_identifier,
        benchmark_identifier=benchmark_identifier,
        conda_active=True)
    os.environ["BS_INSTALL_DEPENDENCIES"] = install_dependence_preference
    assert actual_score == expected_score


@pytest.mark.travis_slow
@pytest.mark.parametrize(
    "model_identifier, benchmark_identifier, expected_score, install_dependencies",
    [
        ("randomembedding-100", "Pereira2018.243sentences-linear",
         approx(0.0285022, abs=_SCORE_ATOL), "newenv"),
        ("randomembedding-100", "Pereira2018.243sentences-linear",
         approx(0.0285022, abs=_SCORE_ATOL), "yes"),
        ("randomembedding-100", "Pereira2018.243sentences-linear",
         approx(0.0285022, abs=_SCORE_ATOL), "no"),
    ]
)
def test_score_with_install_dependencies_slow(
        model_identifier, benchmark_identifier, expected_score, install_dependencies):
    install_dependence_preference = os.environ.get(
        "BS_INSTALL_DEPENDENCIES", "yes")
    os.environ["BS_INSTALL_DEPENDENCIES"] = install_dependencies
    actual_score = score(
        model_identifier=model_identifier,
        benchmark_identifier=benchmark_identifier,
        conda_active=True)
    os.environ["BS_INSTALL_DEPENDENCIES"] = install_dependence_preference
    assert actual_score == expected_score


@pytest.mark.travis_slow
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
