from brainscore_language import load_metric


class TestMetric:
    def test_all_correct(self):
        predictions = targets = ["the", "quick", "brown", "fox", "jumped", "over"]
        metric = load_metric('accuracy')
        score = metric(predictions, targets)
        assert score == 1

    def test_half_correct(self):
        targets = ["the", "quick", "brown", "fox", "jumped", "over"]
        predictions = ["the", "slow", "brown", "cat", "jumped", "under"]
        metric = load_metric('accuracy')
        score = metric(predictions, targets)
        assert score == 0.5

    def test_none_correct(self):
        targets = ["the", "quick", "brown", "fox", "jumped", "over"]
        predictions = ["a", "slow", "white", "cat", "crawled", "under"]
        metric = load_metric('accuracy')
        score = metric(predictions, targets)
        assert score == 0
