import numpy as np
from pytest import approx

from brainscore_language import load_dataset


class TestData:
    def test_language(self):
        assembly = load_dataset('Pereira2018.language')
        assert set(assembly['experiment'].values) == {'243sentences', '384sentences'}
        assert len(assembly['presentation']) == 243 + 384
        assert len(set(assembly['stimulus'].values)) == 243 + 384
        assert 'The concert pianist went blind in adulthood.' in assembly['stimulus'].values
        assert len(set(assembly['subject'].values)) == 10
        assert len(set(assembly['neuroid_id'].values)) == 13553
        assert np.nansum(assembly.values) == approx(1935595.263162177)

    def test_auditory(self):
        assembly = load_dataset('Pereira2018.auditory')
        assert set(assembly['experiment'].values) == {'243sentences', '384sentences'}
        assert len(assembly['presentation']) == 243 + 384
        assert len(set(assembly['stimulus'].values)) == 243 + 384
        assert 'The concert pianist went blind in adulthood.' in assembly['stimulus'].values
        assert len(set(assembly['subject'].values)) == 10
        assert len(set(assembly['neuroid_id'].values)) == 5692
        assert np.nansum(assembly.values) == approx(-257124.1144940494)
