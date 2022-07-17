from enum import Enum

class ArtificialSubject:
    def __init__(self):
        self.UniqueID = "huggingface://gpt2"
    # TODO @Dhaval, @Jim: how do we specify this more accurately for what exactly the outputs are expected to be?
    #  Just more documentation? Also double-check with PIs that these are sufficient for a first round
    Task = Enum('Task', " ".join(['next_word',
                                  'representation',
                                  'surprisal'
                                  ]))
    """
    task to perform
    """
    def identifier(self) -> str:
        """
        The unique identifier for this model.
        :return: e.g. 'glove', or 'distilgpt2'
        """
        #  raise NotImplementedError()
        return self.UniqueID

    def digest_text(self, todostimuli: str):
        raise NotImplementedError()

    # TODO @Dhaval, @Jim, @EvLab: conceptual decision on how we want layer-to-region commitments to happen in the
    #  standard wrapper -- search for best layer on public data?
    def get_representations(self):
        raise NotImplementedError()

    def perform_task(self, task: Task):
        raise NotImplementedError()
