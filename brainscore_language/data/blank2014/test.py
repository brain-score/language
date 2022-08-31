import numpy as np

from brainscore_language import load_dataset


def test_data():
    assembly = load_dataset('Blank2014.fROI')
    assert len(assembly['presentation']) == 1317
    assert "".join(
        assembly.sel(story='Boar').sortby(['sentence_number_in_story', 'part_number_in_sentence'])
        ['stimulus'].values[:10]) == "If you were to journey to the North of England, you would come to a " \
                                     "valley that is surrounded by moors as high as mountains. It is in " \
                                     "this valley where you would find the city of Bradford, where once a " \
                                     "thousand spinning jennies that hummed and clattered spun wool into " \
                                     "money for the long-bearded mill owners. That all mill owners were " \
                                     "generally busy as beavers "
    assert 'Once upon a ' in assembly['stimulus'].values
    assert len(assembly['neuroid']) == 60
    assert len(set(assembly['stimulus_id'].values)) == len(assembly['presentation']), "not all stimulus_ids unique"
    assert set(assembly['story'].values) == {'Aqua', 'Boar', 'Elvis', 'HighSchool',
                                             'KingOfBirds', 'MatchstickSeller', 'MrSticky', 'Tulips'}
    assert set(assembly['subject_id'].values) == {'090', '061', '085', '088', '098'}
    assert set(assembly['fROI_area'].values) == {'10_RH_IFGorb', '11_RH_MFG', '03_LH_IFG', '09_RH_IFG',
                                                 '01_LH_PostTemp', '12_RH_AngG', '07_RH_PostTemp', '04_LH_IFGorb',
                                                 '06_LH_AngG', '02_LH_AntTemp', '08_RH_AntTemp', '05_LH_MFG'}

    mean_assembly = assembly.groupby('subject_id').mean()
    assert not np.isnan(mean_assembly).any()
