from enum import Enum

from brainscore_core import DataAssembly


class ArtificialSubject:
    def identifier(self) -> str:
        """
        The unique identifier for this model.
        :return: e.g. 'glove', or 'distilgpt2'
        """
        raise NotImplementedError()

    Task = Enum('Task', " ".join(['next_word', 'surprisal']))
    """
    task to perform
    """

    def perform_task(self, task: Task):
        raise NotImplementedError()

    RecordingTarget = Enum('RecordingTarget', " ".join([
        'language_system',
        'language_system_left_hemisphere',
        'language_system_right_hemisphere',
    ]))
    """
    location to record from
    """

    RecordingType = Enum('RecordingTarget', " ".join([
        'exact', 'fMRI',
    ]))
    """
    method of recording
    """

    def get_representations(self, recording_target: RecordingTarget, recording_type: RecordingType):
        raise NotImplementedError()

    def digest_text(self, stimuli) -> DataAssembly:
        raise NotImplementedError()
