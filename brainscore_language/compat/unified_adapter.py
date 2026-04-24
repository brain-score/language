"""
Adapter wrapping legacy language ArtificialSubject to conform to UnifiedModel.

process() delegates to the legacy model's digest_text(). Text is extracted
from the StimulusSet's text column. The Dict[str, Assembly] return from
digest_text() is translated to a single DataAssembly based on the current
measurement configuration.
"""

from typing import Any, Dict, Optional, Set

from brainscore_core.model_interface import UnifiedModel, TaskContext


class LanguageModelAdapter(UnifiedModel):

    def __init__(self, legacy_model):
        self._legacy = legacy_model
        self._task_context: Optional[TaskContext] = None
        self._recording_active: bool = False
        self._task_active: bool = False

    @property
    def identifier(self) -> str:
        # Language's identifier is a method, not a property.
        # But load_model() may have overwritten it with a string attribute.
        id_val = self._legacy.identifier
        if callable(id_val):
            return id_val()
        return id_val

    @property
    def region_layer_map(self) -> Dict[str, str]:
        if hasattr(self._legacy, 'region_layer_mapping'):
            return dict(self._legacy.region_layer_mapping)
        return {}

    @property
    def supported_modalities(self) -> Set[str]:
        return {'text'}

    @property
    def required_modalities(self) -> Set[str]:
        # Legacy language models are unimodal — pure text backbones. Hard-
        # require text so pre-flight rejects pairings against a benchmark
        # that does not provide text stimuli.
        return {'text'}

    def process(self, stimuli) -> Any:
        # Extract text from StimulusSet
        if hasattr(stimuli, 'columns'):
            if 'sentence' in stimuli.columns:
                text = list(stimuli['sentence'].values)
            elif 'text' in stimuli.columns:
                text = list(stimuli['text'].values)
            else:
                text = stimuli
        else:
            text = stimuli

        result = self._legacy.digest_text(text)

        # digest_text returns Dict[str, Assembly]. Extract the right key
        # based on which measurement was configured.
        if self._recording_active and 'neural' in result:
            return result['neural']
        if self._task_active and 'behavior' in result:
            return result['behavior']
        # Fallback: return whichever key exists
        if len(result) == 1:
            return next(iter(result.values()))
        return result

    # Backwards-compatible: existing benchmarks call digest_text() directly
    def digest_text(self, text):
        return self._legacy.digest_text(text)

    def start_task(self, task_context: TaskContext) -> None:
        self._task_context = task_context
        self._task_active = True
        # Legacy ArtificialSubject.start_behavioral_task(task) takes ONE arg
        self._legacy.start_behavioral_task(task_context.task_type)

    # Backwards-compatible: existing benchmarks call these directly
    def start_behavioral_task(self, task):
        self._task_active = True
        self._legacy.start_behavioral_task(task)

    def start_neural_recording(self, recording_target, recording_type='fMRI'):
        self._recording_active = True
        self._legacy.start_neural_recording(recording_target, recording_type)

    def start_recording(self, recording_target: str,
                        time_bins=None, recording_type=None, **kwargs) -> None:
        self._recording_active = True
        # Legacy ArtificialSubject.start_neural_recording(target, recording_type)
        # Default recording_type for language is 'fMRI'
        self._legacy.start_neural_recording(
            recording_target, recording_type or 'fMRI'
        )

    def reset(self) -> None:
        self._task_context = None
        self._recording_active = False
        self._task_active = False
        if hasattr(self._legacy, 'current_tokens'):
            self._legacy.current_tokens = None

    def __getattr__(self, name):
        # Delegate attribute access to the legacy model for backwards compatibility
        return getattr(self._legacy, name)
