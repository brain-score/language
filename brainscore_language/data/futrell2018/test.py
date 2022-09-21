import numpy as np

from brainscore_language import load_data


class TestData:
    def test_shape(self):
        assembly = load_data('Futrell2018')

        assert len(assembly['word']) == 10256
        assert len(set(assembly['stimulus_id'].values)) == len(assembly['presentation'])
        assert len(set(assembly['story_id'].values)) == 10
        assert len(set(assembly['sentence_id'].values)) == 481
        assert len(set(assembly['subject_id'].values)) == 180

        mean_assembly = assembly.mean('subject')
        assert not np.isnan(mean_assembly).any()

        assert assembly.bibtex.startswith('@proceedings')
