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
from brainscore_core.session_view import DeclaredInputSession
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (
    drain_session_inputs,
    drive_behavior_session,
    drive_neural_session,
    measurement_settings,
)


# These legacy registrations expose the same embedding at each recording target,
# without a region-to-layer map. Declare their outputs here, leaving the plugins
# unchanged. They cannot answer behavioral tasks.
_EMBEDDING_OUTPUT_CHANNELS = frozenset({
    'neural:language_system',
    'neural:language_system_left_hemisphere',
    'neural:language_system_right_hemisphere',
})
_LEGACY_OUTPUT_CHANNELS = {
    'randomembedding-100': _EMBEDDING_OUTPUT_CHANNELS,
    'randomembedding-1600': _EMBEDDING_OUTPUT_CHANNELS,
    'glove-840b': _EMBEDDING_OUTPUT_CHANNELS,
}


class LanguageModelAdapter(UnifiedModel):
    """Adapt a legacy subject, with explicit outputs when no layer map exists.

    Known embedding registrations have declarations above. Custom legacy subjects
    can supply ``out_channels``; otherwise the public layer map and behavioral
    compatibility path supply the existing defaults.
    """

    def __init__(self, legacy_model: Any, *,
                 out_channels: Optional[Set[str]] = None) -> None:
        self._legacy = legacy_model
        self._declared_out_channels = (
            frozenset(out_channels) if out_channels is not None
            else _LEGACY_OUTPUT_CHANNELS.get(self.identifier)
        )
        self._task_context: Optional[TaskContext] = None
        self._recording_active: bool = False
        self._task_active: bool = False
        # Setup the legacy model currently has. Legacy subjects append every
        # recording registration, so a configuration is applied once until
        # it changes or the adapter is reset.
        self._registered_recordings: set = set()
        self._configured_task = None

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

    @property
    def out_channels(self) -> Set[str]:
        if self._declared_out_channels is not None:
            return set(self._declared_out_channels)
        return super().out_channels | {'behavior'}

    def participate(self, session) -> None:
        """Run a session; a legacy-benchmark session replays the exact ``digest_text``.

        A session carrying ``legacy_text`` (from the legacy-benchmark facade)
        configures the legacy model as the benchmark did and makes one
        ``digest_text`` call that can return neural and behavioral outputs
        together. Other sessions use the typed drivers over ``process``.
        """
        session = DeclaredInputSession(session, self.in_channels)
        channels = tuple(session.requested_output_channels or ())
        missing = set(channels) - self.out_channels
        if not channels or missing:
            raise ValueError(f'Undeclared output channels {sorted(missing)}; declared: '
                             f'{sorted(self.out_channels)}')
        context = session.task_context
        settings = measurement_settings(session)
        if 'legacy_text' not in settings:
            if channels == ('behavior',):
                drive_behavior_session(self, session)
            else:
                drive_neural_session(self, session)
            return
        if context is not None and context.task_type != 'passive':
            self.start_task(context)
        for channel in channels:
            if channel.startswith('neural:'):
                self.start_neural_recording(channel.split(':', 1)[1],
                                            settings.get('recording_type', 'fMRI'))
        drain_session_inputs(session)
        result = self._legacy.digest_text(settings['legacy_text'])
        for channel in channels:
            key = 'neural' if channel.startswith('neural:') else 'behavior'
            session.respond(StreamEvent(channel, result[key], None, meta={'driver': 'legacy'}))

    # Backwards-compatible: existing benchmarks call digest_text() directly
    def digest_text(self, text):
        return self._legacy.digest_text(text)

    def start_task(self, task_context: TaskContext) -> None:
        self._task_context = task_context
        # Legacy ArtificialSubject.start_behavioral_task(task) takes ONE arg
        self.start_behavioral_task(task_context.task_type)

    # Backwards-compatible: existing benchmarks call these directly
    def start_behavioral_task(self, task):
        self._task_active = True
        if task == self._configured_task:
            return
        self._legacy.start_behavioral_task(task)
        self._configured_task = task

    def start_neural_recording(self, recording_target, recording_type='fMRI'):
        self._recording_active = True
        registration = (recording_target, recording_type)
        if registration in self._registered_recordings:
            return  # legacy subjects append; the benchmark registered this once
        self._legacy.start_neural_recording(recording_target, recording_type)
        self._registered_recordings.add(registration)

    def start_recording(self, recording_target: str,
                        time_bins=None, recording_type=None, **kwargs) -> None:
        # Legacy ArtificialSubject.start_neural_recording(target, recording_type)
        # Default recording_type for language is 'fMRI'
        self.start_neural_recording(recording_target, recording_type or 'fMRI')

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
        self._registered_recordings = set()
        self._configured_task = None
        if hasattr(self._legacy, 'current_tokens'):
            self._legacy.current_tokens = None

    @property
    def legacy_model(self):
        """The wrapped legacy model, for tools that need its own attributes.

        Unknown attributes are no longer forwarded implicitly: a benchmark
        sees only what the adapter declares (Contract Rules v3, rule 2.1).
        """
        return self._legacy
