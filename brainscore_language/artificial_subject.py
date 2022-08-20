from typing import Dict, List, Union

from brainio.assemblies import DataAssembly


class ArtificialSubject:
    def identifier(self) -> str:
        """
        The unique identifier for this model.

        :return: e.g. `'glove'`, or `'distilgpt2'`
        """
        raise NotImplementedError()

    class Task:
        """ task to perform """
        # ideally we would define these as enums (also for RecordingTarget and RecordingType)
        # but accessing enums gives weird Enum objects whereas this setup just gives a string
        next_word = 'next_word'
        """ 
        Predict the next word from the preceding context. Output a :class:`~brainio.assemblies.BehavioralAssembly` with 
        next-word predictions as the values and preceding context in the `context` coordinate.
        
        Example:
        
        Calling `digest_text(text = 'the quick brown')` could output 
        
        .. code-block:: python
        
           {'behavior': <xarray.BehavioralAssembly (presentation: 1)>
                        array(['fox']), # the actual next word
                        Coordinates:
                          * presentation  (presentation) MultiIndex
                          - context       (presentation) object 'the quick brown'
                          - stimulus_id   (presentation) int64 0}
        
        Example:
        
        Calling `digest_text(text = ['the quick brown', 'fox jumps', 'over the lazy'])` could output 
        
        .. code-block:: python
        
           {'behavior': <xarray.BehavioralAssembly (presentation: 3)>
                        array(['fox', 'over', 'dog']), # the actual next words
                        Coordinates:
                          * presentation  (presentation) MultiIndex
                          - context       (presentation) object 'the quick brown' 'fox jumps', 'over the lazy'
                          - stimulus_id   (presentation) int64 0 1 2}
        """

        reading_times = 'reading_times'
        """ 
        Output how long it took to read a given text, in . Output a :class:`~brainio.assemblies.BehavioralAssembly` with 
        reading times in milliseconds as the values and text in the `context` coordinate.
        
        Example:
        
        Calling `digest_text(text = 'the quick brown')` could output 
        
        .. code-block:: python
        
           {'behavior': <xarray.BehavioralAssembly (presentation: 1)>
                        array([329.15]), # reading time in milliseconds
                        Coordinates:
                          * presentation  (presentation) MultiIndex
                          - context       (presentation) object 'the quick brown'
                          - stimulus_id   (presentation) int64 0}
        
        Example:
        
        Calling `digest_text(text = ['the quick brown', 'fox jumps', 'over the lazy'])` could output 
        
        .. code-block:: python
        
           {'behavior': <xarray.BehavioralAssembly (presentation: 3)>
                        array([329.15, 337.53, 341.13]), # reading times in milliseconds
                        Coordinates:
                          * presentation  (presentation) MultiIndex
                          - context       (presentation) object 'the quick brown' 'fox jumps', 'over the lazy'
                          - stimulus_id   (presentation) int64 0 1 2}
        """

    def perform_behavioral_task(self, task: Task):
        """
        Instruct the subject to output behavior when queried with
        :meth:`~brainscore_language.artificial_subject.ArtificialSubject.digest_text`. Calling
        :meth:`~brainscore_language.artificial_subject.ArtificialSubject.digest_text` will then output a
        :class:`~brainio.assemblies.BehavioralAssembly` in the `'behavior'` part of the output dict.

        :param task: Which :class:`~brainscore_language.artificial_subject.ArtificialSubject.Task` to use.
        """
        raise NotImplementedError()

    class RecordingTarget:
        """ location to record from """
        language_system = "language_system"
        language_system_left_hemisphere = "language_system_left_hemisphere"
        language_system_right_hemisphere = "language_system_right_hemisphere"

    class RecordingType:
        """ method of recording """

        fMRI = "fMRI"
        """ functional magnetic resonance imaging """

    def perform_neural_recording(self, recording_target: RecordingTarget, recording_type: RecordingType):
        """
        Begin neural recordings from this subject. Calling
        :meth:`~brainscore_language.artificial_subject.ArtificialSubject.digest_text` will then output a
        :class:`~brainio.assemblies.NeuroidAssembly` in the `'neural'` part of the output dict.

        :param recording_target: where in the language-involved parts of the brain to record from.
        :param recording_type: What recording method should be used. Currently only exact unit-level spike-rates are
            available, but this could be e.g. fMRI in the future.
        """
        raise NotImplementedError()

    def digest_text(self, text: Union[str, List[str]]) -> Dict[str, DataAssembly]:
        """
        :param text: text to pass to the subject, either a single string (e.g. `"the quick brown fox jumped"`),
            or a list of strings (e.g. `["the quick", "brown fox jumped"]`)
        :return: a dictionary mapping from `'behavior'` and `'neural'` to a :class:`~brainio.assemblies.DataAssembly`,
            if the subject was instructed to
            :meth:`~brainscore_language.artificial_subject.ArtificialSubject.perform_behavioral_task` and/or
            :meth:`~brainscore_language.artificial_subject.ArtificialSubject.perform_neural_recording` respectively.
            The :class:`~brainio.assemblies.DataAssembly` always contains a `presentation` dimension, corresponding to
            the presentation of text stimuli (including a `context` coordinate for the text,
            and a `stimulus_id` coordinate that uniquely identifies each context).
            The `behavior` output has no other dimension, while the `neural` output has a `neuroid` dimension
            (including a `neuroid_id` coordinate to uniquely identify each recording unit), and sometimes a
            `time_bin` dimension (including `time_bin_start` and `time_bin_end`).
        """
        raise NotImplementedError()
