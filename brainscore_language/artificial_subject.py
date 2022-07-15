from typing import Dict, List, Union

from brainscore_core import DataAssembly


class ArtificialSubject:
    def identifier(self) -> str:
        """
        The unique identifier for this model.
        :return: e.g. 'glove', or 'distilgpt2'
        """
        raise NotImplementedError()

    class Task:
        """ task to perform """
        # ideally we would define these as enums (also for RecordingTarget and RecordingType)
        # but accessing enums gives weird Enum objects
        # whereas this setup just gives a string
        next_word = 'next_word'

    def perform_behavioral_task(self, task: Task):
        raise NotImplementedError()

    class RecordingTarget:
        """ location to record from """
        language_system = "language_system"
        language_system_left_hemisphere = "language_system_left_hemisphere"
        language_system_right_hemisphere = "language_system_right_hemisphere"

    class RecordingType:
        """ method of recording """
        spikerate_exact = "spikerate_exact"  # the exact spike-rate activity of each neuron

    def perform_neural_recording(self, recording_target: RecordingTarget, recording_type: RecordingType):
        raise NotImplementedError()

    def digest_text(self, text: Union[str, List[str]]) -> Dict[str, DataAssembly]:
        """
        :param text: text to pass to the subject, either a single string (e.g. `"the quick brown fox jumped"`),
            or a list of strings (e.g. `["the quick", "brown fox jumped"]`)
        :return: a dictionary mapping from `'behavior'` and `'neural'` to an assembly, if the subject was instructed to
            `perform_task` and/or `get_representations` respectively.
        """
        raise NotImplementedError()
