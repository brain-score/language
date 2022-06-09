from enum import Enum


class BrainModel:
    # TODO @EvLab: do these make sense?
    RecordingTarget = Enum('RecordingTarget', " ".join([
        'language_system',
        'language_system_left_hemisphere',
        'language_system_right_hemisphere',
    ]))
    """
    location to record from
    """

    # TODO @Dhaval, @Jim: how do we specify this more accurately for what exactly the outputs are expected to be?
    #  Just more documentation? Also double-check with PIs that these are sufficient for a first round
    Task = Enum('Task', " ".join(['next_word', 'surprisal']))
    """
    task to perform
    """

    def digest_text(self, todostimuli):
        raise NotImplementedError()

    def start_recording(self, recording_target: RecordingTarget):
        raise NotImplementedError()

    def start_task(self, task: Task):
        raise NotImplementedError()
