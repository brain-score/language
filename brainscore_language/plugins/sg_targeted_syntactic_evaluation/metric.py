import numpy as np

from brainscore_core.metrics import Score, Metric
from brainscore_language import metrics


class Accuracy(Metric):
    """
    Standard top-1 accuracy for e.g. next-word performance of model predictions.
    """

    def __call__(self, predictions, targets):
        correct = np.array(predictions) == np.array(targets)
        score = Score(np.mean(correct))
        score.attrs['raw'] = correct
        return score


metrics['accuracy'] = Accuracy
