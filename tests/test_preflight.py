"""Tests for pre-flight checks wired into the language scoring pipeline."""

import math

import pytest
from unittest.mock import patch, MagicMock

from brainscore_core.compatibility import CompatibilityError, ExecutionNotFeasibleError
from brainscore_core.memory import MemoryError
from brainscore_core.contract import Subject


class _Subject(Subject):
    """A small real subject: it has what it declares and nothing else.

    A MagicMock answers every attribute, so a pre-flight that starts to read a
    new declaration would pass on mock values; this subject does not.
    """
    identifier = 'test-model'
    in_channels = frozenset()
    out_channels = frozenset()
    required_channels = frozenset()
    region_layer_map = {}

    def __init__(self, in_channels):
        self.in_channels = frozenset(in_channels)

    def participate(self, session):
        raise AssertionError('the stand-in benchmark never runs a session')


class TestPreflightInRunScore:
    """Verify that _run_score calls check_compatibility and check_memory."""

    def _make_model_and_benchmark(self, model_in_channels, bench_required):
        """A real subject and a stand-in benchmark for pre-flight testing."""
        model = _Subject(model_in_channels)

        benchmark = MagicMock(spec=['identifier', 'required_modalities', '__call__'])
        benchmark.identifier = 'test-bench'
        benchmark.required_modalities = bench_required
        return model, benchmark

    @patch('brainscore_language.load_benchmark')
    @patch('brainscore_language.load_model')
    def test_incompatible_model_returns_na_before_compute(
        self, mock_load_model, mock_load_benchmark
    ):
        from brainscore_language import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'vision:image'},
            bench_required={'text'},
        )
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        result = _run_score('test-model', 'test-bench', check_mem=False)

        assert math.isnan(float(result))
        assert result.attrs['status'] == 'N/A'
        assert "takes in none of the inputs" in result.attrs['reason']
        assert result.attrs['model_identifier'] == 'test-model'
        assert result.attrs['benchmark_identifier'] == 'test-bench'
        benchmark.assert_not_called()

    @patch('brainscore_language.load_benchmark')
    @patch('brainscore_language.load_model')
    def test_score_metadata_attrs_are_stamped(
        self, mock_load_model, mock_load_benchmark
    ):
        from brainscore_language import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'text'},
            bench_required={'text'},
        )
        model.out_channels = {'behavior'}
        benchmark.parent = 'behavior'
        score = MagicMock()
        score.attrs = {'existing': 'preserved'}
        benchmark.return_value = score
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        result = _run_score('test-model', 'test-bench', check_mem=False)

        benchmark.assert_called_once_with(model)   # a compatible pair runs once
        assert result is score
        assert score.attrs['existing'] == 'preserved'
        assert score.attrs['in_channels'] == ('text',)
        assert score.attrs['out_channels'] == ('behavior',)
        assert score.attrs['protocol'] == 'behavior:readout'
        assert score.attrs['subject_id'] == 'test-model'
        assert score.attrs['compatibility']['match'] == 'full'
        assert score.attrs['harness_id'] == 'brainscore_language'
        # No session answered in this fake run; the plan and versions are still recorded.
        assert score.attrs['delivery'] is None
        assert score.attrs['regions_per_pass'] is None   # no subject said which plan it used
        assert score.attrs['software']['python']
        assert 'brainscore_core' in score.attrs['code_commits']

    @patch('brainscore_language.load_benchmark')
    @patch('brainscore_language.load_model')
    def test_behavior_protocol_comes_from_recorded_measures(
            self, mock_load_model, mock_load_benchmark):
        from brainscore_language import _run_score
        from brainscore_core.streaming import Session, StreamEvent
        from brainscore_core.supported_data_standards.brainio.assemblies import (
            BehavioralAssembly,
        )

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'text'},
            bench_required={'text'},
        )
        model.out_channels = {'behavior'}
        benchmark.parent = 'behavior'
        benchmark.requested_output_channels = {'behavior'}
        score = MagicMock()
        score.attrs = {}

        def run(candidate):
            # A subject's behavioral response, accepted by a session.
            Session((), ['behavior']).respond(
                StreamEvent('behavior', BehavioralAssembly(
                    [[1.0, 0.0]], dims=['presentation', 'choice'],
                    attrs={'measure': {'kind': 'reported', 'source': 'reply'}}), None))
            return score

        benchmark.side_effect = run
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        _run_score('test-model', 'test-bench', check_mem=False)

        assert score.attrs['protocol'] == 'behavior:generation'
        assert score.attrs['measure'] == 'reported'

    def test_an_older_core_fails_with_a_clear_message(self):
        from brainscore_language import _run_score
        import brainscore_core.score_metadata as score_metadata

        saved = score_metadata.record_measures
        del score_metadata.record_measures
        try:
            with pytest.raises(ImportError, match='core ad7288a or later'):
                _run_score('test-model', 'test-bench', check_mem=False)
        finally:
            score_metadata.record_measures = saved

    @patch('brainscore_language.load_benchmark')
    @patch('brainscore_language.load_model')
    def test_channel_mismatch_names_the_channel_in_the_na_result(
        self, mock_load_model, mock_load_benchmark
    ):
        from brainscore_language import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'text'},
            bench_required={'text'},
        )
        benchmark.required_input_channels = {'vision:image'}
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        result = _run_score('test-model', 'test-bench', check_mem=False)

        assert result.attrs['status'] == 'N/A'
        assert "vision" in result.attrs['reason']
        assert result.attrs['missing_channels'] == sorted(benchmark.required_input_channels)

        benchmark.assert_not_called()

    @patch('brainscore_language.load_benchmark')
    @patch('brainscore_language.load_model')
    def test_a_refusal_at_session_start_is_an_error(
        self, mock_load_model, mock_load_benchmark
    ):
        from brainscore_language import _run_score
        from brainscore_core.streaming import Session

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'text'},
            bench_required={'text'},
        )
        model.out_channels = {'behavior'}
        benchmark.requested_output_channels = {'behavior'}

        def refuse(session):
            raise ExecutionNotFeasibleError('no one part answers', channels=session.outputs)
        model.participate = refuse
        benchmark.side_effect = lambda candidate: candidate.participate(Session([], ['behavior']))
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        with pytest.raises(ExecutionNotFeasibleError, match='no one part answers') as caught:
            _run_score('test-model', 'test-bench', check_mem=False)

        assert not isinstance(caught.value, CompatibilityError)   # an error, never N/A

        assert caught.value.channels == set(['behavior'])

    @patch('brainscore_core.memory.check_memory')
    @patch('brainscore_language.load_benchmark')
    @patch('brainscore_language.load_model')
    def test_check_mem_false_skips_memory_check(
        self, mock_load_model, mock_load_benchmark, mock_check_memory
    ):
        from brainscore_language import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'text'},
            bench_required={'text'},
        )
        score = MagicMock()
        score.attrs = {}
        benchmark.return_value = score
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark

        _run_score('test-model', 'test-bench', check_mem=False)
        mock_check_memory.assert_not_called()

    @patch('brainscore_core.memory.check_memory')
    @patch('brainscore_language.load_benchmark')
    @patch('brainscore_language.load_model')
    def test_memory_error_propagates(
        self, mock_load_model, mock_load_benchmark, mock_check_memory
    ):
        from brainscore_language import _run_score

        model, benchmark = self._make_model_and_benchmark(
            model_in_channels={'text'},
            bench_required={'text'},
        )
        mock_load_model.return_value = model
        mock_load_benchmark.return_value = benchmark
        mock_check_memory.side_effect = MemoryError("not enough memory")

        with pytest.raises(MemoryError, match="not enough memory"):
            _run_score('test-model', 'test-bench', check_mem=True)


class _NeuralOnly(Subject):
    """A neural-only subject that records each input it is given."""
    identifier = 'neural-only'
    in_channels = frozenset({'text'})
    out_channels = frozenset({'neural:language_network'})
    required_channels = frozenset()

    def __init__(self):
        self.inputs = []

    def participate(self, session):
        while (event := session.next_input()) is not None:
            self.inputs.append(event)


class TestRefusalDuringTheRunIsAnError:
    """N/A comes only from the declarations; a refusal during the run raises."""

    @staticmethod
    def _legacy_benchmark(run):
        from brainscore_core.legacy_facade import mark_legacy_benchmark
        benchmark = MagicMock(spec=['identifier', 'required_modalities', '__call__'])
        benchmark.identifier = 'legacy-reading-times'
        benchmark.required_modalities = {'text'}
        benchmark.side_effect = run
        return mark_legacy_benchmark(benchmark)

    @patch('brainscore_language.load_benchmark')
    @patch('brainscore_language.load_model')
    def test_a_legacy_behavioral_task_for_a_neural_only_subject_is_an_error(
            self, mock_load_model, mock_load_benchmark):
        from brainscore_language import _run_score
        subject = _NeuralOnly()

        def run(candidate):       # as Futrell2018: the task first, then the text
            candidate.start_behavioral_task('reading_times')
            candidate.digest_text(['the', 'cat'])
        mock_load_model.return_value = subject
        mock_load_benchmark.return_value = self._legacy_benchmark(run)

        with pytest.raises(ExecutionNotFeasibleError, match="declares no 'behavior' output") as caught:
            _run_score('neural-only', 'legacy-reading-times', check_mem=False)

        assert caught.value.channels == {'behavior'}
        assert subject.inputs == []

    @patch('brainscore_language.load_benchmark')
    @patch('brainscore_language.load_model')
    def test_a_refusal_after_an_earlier_session_computed_is_an_error(
            self, mock_load_model, mock_load_benchmark):
        from brainscore_language import _run_score
        subject = _NeuralOnly()

        def run(candidate):       # a neural measurement, then a behavioral task
            candidate.start_neural_recording('language_network', 'fMRI')
            try:
                candidate.digest_text(['the', 'cat'])
            except Exception:
                pass                # this fake subject gives no assembly back
            candidate.start_behavioral_task('reading_times')
        mock_load_model.return_value = subject
        mock_load_benchmark.return_value = self._legacy_benchmark(run)

        with pytest.raises(ExecutionNotFeasibleError, match="declares no 'behavior' output"):
            _run_score('neural-only', 'legacy-reading-times', check_mem=False)
        assert subject.inputs

    @patch('brainscore_language.load_benchmark')
    @patch('brainscore_language.load_model')
    def test_a_runtime_error_passes_through(self, mock_load_model, mock_load_benchmark):
        from brainscore_language import _run_score

        def crash(candidate):
            raise RuntimeError('model crashed')
        mock_load_model.return_value = _NeuralOnly()
        mock_load_benchmark.return_value = self._legacy_benchmark(crash)

        with pytest.raises(RuntimeError, match='model crashed'):
            _run_score('neural-only', 'legacy-reading-times', check_mem=False)

    @patch('brainscore_language.load_benchmark')
    @patch('brainscore_language.load_model')
    def test_a_malformed_benchmark_declaration_is_an_error(self, mock_load_model, mock_load_benchmark):
        from brainscore_language import _run_score
        benchmark = self._legacy_benchmark(lambda candidate: None)
        benchmark.in_channels = {'text'}          # a field benchmarks do not read
        del benchmark.required_modalities
        mock_load_model.return_value = _NeuralOnly()
        mock_load_benchmark.return_value = benchmark

        with pytest.raises(ValueError, match='Use required_input_channels'):
            _run_score('neural-only', 'legacy-reading-times', check_mem=False)
        benchmark.assert_not_called()


class _LeavesALesion:
    """A benchmark that applies an intervention and, unless told to, leaves it active."""
    identifier = 'leaves-a-lesion'
    version = 1
    required_input_channels = {'text'}
    requested_output_channels = {'perturbation'}

    def __init__(self, undo):
        self.undo = undo

    def __call__(self, subject):
        from brainscore_core.events import Perturbation, Selection, StateChange
        from brainscore_core.metrics import Score
        from brainscore_core.streaming_helpers import apply_state_change
        applied = apply_state_change(subject, StateChange(
            kind='ablation', target=Selection(layer='x'), perturbation=Perturbation(kind='zero')))
        if self.undo:
            apply_state_change(subject, StateChange(kind='reset', handle_id=applied.handle_id))
        return Score(1.0)


class _LesionSubject(Subject):
    identifier = 'lesion-subject'
    in_channels = frozenset({'text'})
    out_channels = frozenset({'perturbation'})

    def participate(self, session):
        from brainscore_core.events import PerturbationApplied
        from brainscore_core.streaming import StreamEvent
        change = session.next_input().payload
        session.respond(StreamEvent(channel='perturbation', t_ms=0.0, payload=PerturbationApplied(
            change.handle_id or 'h1', change.target, change.perturbation)))


@pytest.mark.parametrize('undo', [False, True])
def test_a_benchmark_that_leaves_an_intervention_active_is_an_error(undo):
    """The release check of brainscore.score runs here too, before a result is returned."""
    from brainscore_language import _run_score
    with patch('brainscore_language.load_benchmark', return_value=_LeavesALesion(undo)), \
            patch('brainscore_language.load_model', return_value=_LesionSubject()):
        if undo:
            assert float(_run_score('lesion-subject', 'leaves-a-lesion', check_mem=False)) == 1.0
        else:
            with pytest.raises(RuntimeError, match=r"left intervention handle\(s\) \['h1'\]"):
                _run_score('lesion-subject', 'leaves-a-lesion', check_mem=False)
