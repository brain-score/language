"""
Adapter wrapping legacy language ArtificialSubject to conform to the unified
Subject interface (formerly UnifiedModel).

process() delegates to the legacy model's digest_text(). Text is extracted
from the StimulusSet's text column. The Dict[str, Assembly] return from
digest_text() is translated to a single DataAssembly based on the current
measurement configuration.
"""

from typing import Any, Dict, Optional, Set

from brainscore_core.model_interface import UnifiedModel, TaskContext
from brainscore_core.streaming_helpers import _drive_neural_session_via_process


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
        import pandas as pd
        if isinstance(stimuli, pd.DataFrame):
            return self._process_table(stimuli)
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
        return self._select_output(result)

    def _select_output(self, result):
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

    def _process_table(self, stimuli):
        import numpy as np
        import xarray as xr
        from brainscore_core.text import text_parts, text_groups
        from brainscore_core.supported_data_standards.brainio.assemblies import walk_coords

        if 'stimulus_id' not in stimuli.columns or len(stimuli) == 0:
            raise ValueError('Text StimulusSet requires stimulus_id and at least one row.')
        parts = text_parts(stimuli)
        outputs, positions_out = [], []
        for positions in text_groups(stimuli):
            group_parts = [parts[i] for i in positions]
            output = self._select_output(self._legacy.digest_text(group_parts))
            if ('presentation' not in getattr(output, 'dims', ())
                    or output.sizes['presentation'] != len(positions)):
                raise ValueError('Legacy language output must contain one presentation per input part.')
            # Legacy outputs promise input order; part_number, when present,
            # supplies an explicit correspondence that also detects bad rows.
            try:
                part_numbers = np.asarray(output['part_number'].values)
            except KeyError:
                part_numbers = None
            if part_numbers is not None:
                if (part_numbers.ndim != 1
                        or not np.array_equal(np.sort(part_numbers), np.arange(len(positions)))):
                    raise ValueError('Legacy part_number must be a permutation of input positions.')
                output = output.isel(presentation=np.argsort(part_numbers))
            try:
                returned_parts = list(output['stimulus'].values)
            except KeyError:
                returned_parts = None
            if returned_parts is not None and returned_parts != group_parts:
                raise ValueError('Legacy stimulus coordinates do not match the input text parts.')
            coords = {name: (dims, values) for name, dims, values in walk_coords(output)}
            for column in stimuli.columns:
                if column in coords and coords[column][0] != ('presentation',):
                    raise ValueError(f'Input metadata {column!r} conflicts with a non-presentation coordinate.')
                coords[column] = ('presentation', list(stimuli.iloc[positions][column]))
            restored = type(output)(output.values, dims=output.dims, coords=coords,
                                    attrs=output.attrs)
            if 'presentation' in restored.indexes:
                restored = restored.reset_index('presentation')
            outputs.append(restored)
            positions_out.extend(positions)
        return xr.concat(outputs, dim='presentation').isel(
            presentation=np.argsort(positions_out))

    def interact(self, session) -> None:
        _drive_neural_session_via_process(self, session)

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
        reset = getattr(self._legacy, 'reset', None)
        if callable(reset):
            reset()
        # Permanent compatibility with legacy helpers without reset(). These
        # are their actual measurement/episode fields, not just adapter flags.
        state = vars(self._legacy)
        for name in ('neural_recordings', '_neural_recordings'):
            if name in state:
                setattr(self._legacy, name, [])
        for name in ('behavioral_task', '_behavioral_task', 'output_to_behavior',
                     '_behavioral_function', 'current_tokens'):
            if name in state:
                setattr(self._legacy, name, None)
        if '_token_count' in state:
            self._legacy._token_count = 0
        self._task_context = None
        self._recording_active = False
        self._task_active = False
        if hasattr(self._legacy, 'current_tokens'):
            self._legacy.current_tokens = None

    def __getattr__(self, name):
        # Delegate attribute access to the legacy model for backwards compatibility
        return getattr(self._legacy, name)
