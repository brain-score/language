import numpy as np
from pytest import approx

from brainscore_language import load_dataset


class TestData:
    def test_Pereira2018(self):
        assembly = load_dataset('Pereira2018.language_system')
        assert set(assembly['experiment'].values) == {'243sentences', '384sentences'}
        assert len(assembly['presentation']) == 243 + 384
        assert len(set(assembly['subject'].values)) == 10
        assert len(set(assembly['neuroid_id'].values)) == 13553
        assert np.nansum(assembly.values) == approx(1935595.263162177)

    def test_Fedorenko2016(self):
        assembly = load_dataset('Fedorenko2016.language')
        assert len(assembly['presentation']) == 416
        assert len(assembly['neuroid']) == 97
        assert len(np.unique(assembly['subject_UID'])) == 5

    def test_Blank2014(self):
        assembly = load_dataset('Blank2014.fROI')
        assert len(assembly['presentation']) == 1317
        assert len(assembly['neuroid']) == 60
        assert len(set(assembly['stimulus_id'].values)) == len(assembly['presentation'])
        assert set(assembly['story'].values) == {'Aqua', 'Boar', 'Elvis', 'HighSchool',
                                                 'KingOfBirds', 'MatchstickSeller', 'MrSticky', 'Tulips'}
        assert set(assembly['subject_id'].values) == {'090', '061', '085', '088', '098'}
        assert set(assembly['fROI_area'].values) == {'10_RH_IFGorb', '11_RH_MFG', '03_LH_IFG', '09_RH_IFG',
                                                     '01_LH_PostTemp', '12_RH_AngG', '07_RH_PostTemp', '04_LH_IFGorb',
                                                     '06_LH_AngG', '02_LH_AntTemp', '08_RH_AntTemp', '05_LH_MFG'}

        mean_assembly = assembly.groupby('subject_id').mean()
        assert not np.isnan(mean_assembly).any()
