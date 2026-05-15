import numpy as np

from brainscore_language import load_dataset


class TestData:
    def test_language(self):
        assembly = load_dataset('Tuckute2024.language')

        assert assembly.dims == ('presentation', 'neuroid')
        assert assembly.shape == (1000, 1)
        assert len(assembly.stimulus.values) == 1000
        assert np.unique(assembly.stimulus.values).shape[0] == 1000
        assert len(assembly.stimulus_id.values) == 1000
        assert np.unique(assembly.stimulus_id.values).shape[0] == 1000
        assert assembly.neuroid_id.values == [1]
        assert 1.28 < np.max(assembly.data) < 1.29
