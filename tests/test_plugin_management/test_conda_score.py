import numpy as np
import os
import pytest
from numpy.random import RandomState
from pathlib import Path
from pytest import approx

from brainio.assemblies import NeuroidAssembly
from brainscore_language import load_metric
from brainscore_language.benchmarks.futrell2018 import Futrell2018Pearsonr
from brainscore_language.plugin_management.conda_score import CondaScore, SCORE_PATH
from brainscore_language.utils.ceiling import ceiling_normalize


def _make_assembly():
    values = RandomState(1).standard_normal(30 * 25).reshape((30, 25))
    assembly = NeuroidAssembly(values,
                               coords={'stimulus_id': ('presentation', np.arange(30)),
                                       'stimulus_category': ('presentation', ['a', 'b', 'c'] * 10),
                                       'neuroid_id': ('neuroid', np.arange(25)),
                                       'region': ('neuroid', ['some_region'] * 25)},
                               dims=['presentation', 'neuroid'])
    return assembly


def _create_dummy_score():
    assembly = _make_assembly()
    metric = load_metric('linear_pearsonr')
    raw_score = metric(assembly1=assembly, assembly2=assembly)
    score = ceiling_normalize(raw_score, Futrell2018Pearsonr().ceiling)
    return score


def test_save_and_read_score():
    output = _create_dummy_score()
    CondaScore.save_score(output)
    assert Path(SCORE_PATH).is_file()
    result = CondaScore.read_score()
    assert not Path(SCORE_PATH).is_file()
    assert output == result


@pytest.mark.memory_intense
def test_score_in_env():
    result = CondaScore(model_identifier='distilgpt2', benchmark_identifier='Futrell2018-pearsonr')
    score = result.score
    assert score == approx(0.3614471, abs=0.001)
