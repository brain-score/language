import numpy as np

from brainscore_language import load_dataset
from brainscore_language.data.tuckute2024.data_packaging import load_tuckute2024_5subj

class TestData:
    def test_language(self,
                      load_from_cache: bool = False):
        if load_from_cache:
            assembly = load_dataset('tuckute2024_5subj_lang_LH_netw')
        else:
            assembly = load_tuckute2024_5subj()

        assert assembly.dims == ('presentation', 'neuroid')
        assert assembly.shape == (1000, 1)
        assert len(assembly.stimulus.values) == 1000
        assert np.unique(assembly.stimulus.values).shape[0] == 1000
        assert len(assembly.stimulus_id.values) == 1000
        assert np.unique(assembly.stimulus_id.values).shape[0] == 1000
        assert assembly.neuroid_id.values == [1]
        assert 1.28 < np.max(assembly.data) < 1.29



