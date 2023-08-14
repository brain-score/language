import copy
import logging

import numpy as np
import pytest
from numpy.random import RandomState
from pytest import approx

from brainio.assemblies import NeuroidAssembly
from brainscore_language import load_dataset, ArtificialSubject, load_benchmark

_logger = logging.getLogger(__name__)


class TestData:
    def test_language(self):
        assembly = load_dataset("Pereira2018_v2022.language")
        assert set(assembly["experiment"].values) == {
            "PereiraE2_96pass",
            "PereiraE3_72pass",
        }
        _logger.info("experiment names match up!")

        assert len(assembly["presentation"]) == 243 + 384
        _logger.info(f'no. of presentation IDs == {len(assembly["presentation"])}')

        assert len(set(assembly["stimuli"].values)) == 243 + 384
        _logger.info(f'no. of stimuli == {len(assembly["stimuli"].values)}')

        assert (
            "The concert pianist went blind in adulthood."
            in assembly["stimuli"].values
        )
        _logger.info(
            f'stimulus "The concert pianist went blind in adulthood." is present!'
        )

        assert len(set(assembly["subject"].values)) == 10
        _logger.info(f'no. of subjects == {len(set(assembly["subject"].values))}')

        assert (
            len(set(assembly["neuroid"].values)) == 120
        )  # potential TODO: rename to neuroid_id
        _logger.info(f'no. of neuroids == {len(set(assembly["neuroid"].values))}')
