from brainscore_language import load_data


class TestData:
    def test_firstline(self):
        data = load_data('wikitext-2/test')
        assert data[1] == ' = Robert Boulter = \n'

    def test_length(self):
        data = load_data('wikitext-2/test')
        assert len(data) == 4358
