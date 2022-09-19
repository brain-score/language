import copy

import numpy as np
import pytest
from numpy.random import RandomState
from pytest import approx

from brainio.assemblies import NeuroidAssembly
from brainscore_language import load_dataset, ArtificialSubject, load_benchmark


class TestBenchmark:
    class DummyModel(ArtificialSubject):
        def __init__(self, neural_activity):
            self.neural_activity = neural_activity

        def digest_text(self, stimuli):
            assert len(stimuli) == len(self.neural_activity["presentation"])
            self.neural_activity["sentence"] = "presentation", stimuli  # copy over
            return {"neural": self.neural_activity}

        def perform_neural_recording(
            self,
            recording_target: ArtificialSubject.RecordingTarget,
            recording_type: ArtificialSubject.RecordingType,
        ):
            assert recording_target == ArtificialSubject.RecordingTarget.language_system
            assert recording_type == ArtificialSubject.RecordingType.fMRI

    @pytest.mark.parametrize(
        "experiment, expected_score",
        [
            (243, approx(0.0017534 / 0.35378928, abs=0.001)),
            (384, approx(0.01215216 / 0.36343748, abs=0.001)),
        ],
    )
    def test_dummy_bad(self, experiment, expected_score):
        benchmark = load_benchmark(
            f"Pereira2018_v2022.{experiment}sentences-linreg_pearsonr"
        )
        neural_activity = RandomState(0).random(
            size=(experiment, 25)
        )  # presentation x neuroid
        neural_activity = NeuroidAssembly(
            neural_activity,
            coords={
                "stimulus_seq": ("presentation", np.arange(experiment)),
                "stimulus_num": ("presentation", np.arange(experiment)),
                "neuroid_id": ("neuroid", np.arange(25)),
                "region": ("neuroid", ["some_region"] * 25),
            },
            dims=["presentation", "neuroid"],
        )
        dummy_model = TestBenchmark.DummyModel(neural_activity=neural_activity)
        score = benchmark(dummy_model)
        assert score == expected_score

    @pytest.mark.parametrize(
        "experiment",
        [
            243,
            384,
        ],
    )
    def test_exact(self, experiment):
        benchmark = load_benchmark(
            f"Pereira2018_v2022.{experiment}sentences-linreg_pearsonr"
        )
        exact_data = copy.deepcopy(benchmark.data).reset_index("presentation")
        del exact_data["presentation"], exact_data["sentence"]
        exact_data = exact_data.set_index(
            presentation=list(exact_data["presentation"].coords)
        )
        dummy_model = TestBenchmark.DummyModel(neural_activity=exact_data)
        score = benchmark(dummy_model)
        print(score)
        assert score == approx(1)

    # @pytest.mark.parametrize(
    #     "experiment, expected_ceiling",
    #     [
    #         (243, 0.35378928),
    #         (384, 0.36343748),
    #     ],
    # )
    # def test_ceiling(self, experiment, expected_ceiling):
    #     benchmark = load_benchmark(f"Pereira2018.{experiment}sentences-linear")
    #     ceiling = benchmark.ceiling
    #     assert ceiling == approx(expected_ceiling, abs=0.0005)

    # @pytest.mark.parametrize("experiment", [243, 384])
    # def test_ceiling_raw(self, experiment):
    #     benchmark = load_benchmark(f"Pereira2018.{experiment}sentences-linear")
    #     ceiling = benchmark.ceiling
    #     assert hasattr(ceiling, "raw")
    #     assert set(ceiling.raw.dims) == {"neuroid"}
    #     assert ceiling.raw.median() == ceiling
    #     assert hasattr(ceiling.raw, "raw")
    #     assert set(ceiling.raw.raw.dims) == {
    #         "sub_subject",
    #         "num_subjects",
    #         "split",
    #         "neuroid",
    #     }
