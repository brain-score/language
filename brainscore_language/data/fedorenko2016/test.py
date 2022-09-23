import numpy as np

from brainscore_language import load_dataset


def test_data():
    assembly = load_dataset('Fedorenko2016.language')
    assert len(assembly['presentation']) == 416
    assert len(set(assembly['stimulus'].values)) == len(set(assembly['word'].values)) == 255
    assert ' '.join(assembly.sel(sentence_id=0)['stimulus'].values) == 'ALEX WAS TIRED SO HE TOOK A NAP'
    assert set(assembly['word_num'].values) == {0, 1, 2, 3, 4, 5, 6, 7}
    assert len(set(assembly['sentence_id'].values)) == 52
    assert len(assembly['neuroid']) == 97
    assert len(np.unique(assembly['subject_UID'])) == 5
