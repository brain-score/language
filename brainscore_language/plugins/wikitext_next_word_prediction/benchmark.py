import logging
import re
import string

from brainscore_core.benchmarks import BenchmarkBase
from brainscore_language import load_dataset, load_metric, benchmarks
from brainscore_language.artificial_subject import ArtificialSubject

logger = logging.getLogger(__name__)

BIBTEX = """@proceedings{merity2017pointer,
  title={Pointer sentinel mixture models},
  author={Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
  conference={International Conference on Learning Representations (ICLR)},
  url={https://openreview.net/forum?id=Byj72udxe},
  year={2016}
}"""


class Merity2016nextwordAccuracy(BenchmarkBase):
    def __init__(self):
        super(Merity2016nextwordAccuracy, self).__init__(
            identifier='Merity2016nextword-accuracy',
            version=1,
            parent='engineering',
            ceiling_func=None,
            bibtex=BIBTEX)  # TODO: I think this should go into the data plugin somehow
        self.data = load_dataset('wikitext-2/test')
        self.metric = load_metric('accuracy')

    def __call__(self, candidate: ArtificialSubject):
        candidate.perform_task(ArtificialSubject.Task.next_word)
        contexts, targets = self.build_contexts()
        predictions = candidate.digest_text(contexts)
        score = self.metric(predictions, targets)
        return score

    def build_contexts(self):
        """
        Create context-target pairs from `self.data`.
        Note that there is *no tokenization* here, because we treat the candidate like a subject.
        This means we split on white spaces and ask for the next *word* within this context
        (rather than the next token).
        :return:
        """
        contexts = []
        targets = []
        previous_context = ''
        page_header = r'^=[^=]*(=\n)$'  # = at beginning and end, but no more
        for line in self.data:
            if re.match(page_header, line):
                previous_context = ''
            line = line.strip()
            whitespace_indices = [i for i, char in enumerate(line) if char in string.whitespace and i > 0]
            # the context is everything within this page up to (but not including) the current word
            line_contexts = [previous_context + line[:index] for index in whitespace_indices[:-1]]
            # the target for each of these contexts is the current word
            # this current implementation also makes the subject predict next "words" like whitespace, commas, or '@-@'
            line_targets = [line[whitespace_indices[indices_index - 1]:whitespace_indices[indices_index]].strip()
                            for indices_index in range(1, len(whitespace_indices))]
            contexts += line_contexts
            targets += line_targets
        assert len(contexts) == len(targets)
        return contexts, targets


benchmarks['Merity2016nextword-accuracy'] = Merity2016nextwordAccuracy
