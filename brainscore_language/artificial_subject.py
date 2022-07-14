from enum import Enum

class ArtificialSubject:
    # TODO @Dhaval, @Jim: how do we specify this more accurately for what exactly the outputs are expected to be?
    #  Just more documentation? Also double-check with PIs that these are sufficient for a first round
    Task = Enum('Task', " ".join(['next_word',
                                  'fill_mask',
                                  ]))
    """
    task to perform
    """

    def identifier(self) -> str:
        """
        The unique identifier for this model.
        :return: e.g. 'glove', or 'distilgpt2'
        """
        raise NotImplementedError()

    def digest_text(self, todostimuli: str):
        raise NotImplementedError()

    def get_representations(self):
        raise NotImplementedError()

    def perform_task(self, task: Task):
        raise NotImplementedError()
