import logging

import numpy as np
import pytest
from pytest import approx

from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

_logger = logging.getLogger(__name__)


class TestNextWord:
    @pytest.mark.parametrize('model_identifier, expected_next_word, bidirectional', [
        pytest.param('bert-base-uncased', 'and', True, marks=pytest.mark.memory_intense),
        pytest.param('bert-base-uncased', '.', False, marks=pytest.mark.memory_intense),
        pytest.param('gpt2-xl', 'jumps', False, marks=pytest.mark.memory_intense),
        ('distilgpt2', 'es', False),
    ])
    def test_single_string(self, model_identifier, expected_next_word, bidirectional):
        """
        This is a simple test that takes in text = 'the quick brown fox', and tests the next word.
        This test is a stand-in prototype to check if our model definitions are correct.
        """

        model = HuggingfaceSubject(model_id=model_identifier, region_layer_mapping={}, bidirectional=bidirectional)
        text = 'the quick brown fox'
        _logger.info(f'Running {model.identifier()} with text "{text}"')
        model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
        next_word = model.digest_text(text)['behavior'].values
        assert next_word == expected_next_word

    @pytest.mark.parametrize('model_identifier, expected_next_words, bidirectional', [
        pytest.param('bert-base-uncased', [';', 'the', 'water'], True, marks=pytest.mark.memory_intense),
        pytest.param('bert-base-uncased', ['.', '.', '.'], False, marks=pytest.mark.memory_intense),
        pytest.param('gpt2-xl', ['jumps', 'the', 'dog'], False, marks=pytest.mark.memory_intense),
        ('distilgpt2', ['es', 'the', 'fox'], False),
    ])
    def test_list_input(self, model_identifier, expected_next_words, bidirectional):
        """
        This is a simple test that takes in text = ['the quick brown fox', 'jumps over', 'the lazy'], and tests the
        next word for each text part in the list.
        This test is a stand-in prototype to check if our model definitions are correct.
        """
        model = HuggingfaceSubject(model_id=model_identifier, region_layer_mapping={}, bidirectional=bidirectional)
        text = ['the quick brown fox', 'jumps over', 'the lazy']
        _logger.info(f'Running {model.identifier()} with text "{text}"')
        model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
        next_words = model.digest_text(text)['behavior']
        np.testing.assert_array_equal(next_words, expected_next_words)

    def test_over_max_length_input(self):
        # max_input_length of distilgpt2 is 1024 tokens. Prompt it with text longer than this length, the model should
        # handle this case gracefully and not fail (e.g. truncate input)
        text = 'lorem ipsum dolor sit amet'.split() * 205
        assert len(text) > 1024
        text = ' '.join(text)
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
        model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
        next_words = model.digest_text(text)['behavior']
        assert len(next_words) == 1

    def test_0_dimensional_tensor(self):
        # Some benchmarks (e.g. Wikitext-accuracy) will incur a 0-dim pred_id
        # ensure that this is handled gracefully
        from brainscore_language import load_benchmark
        benchmark = load_benchmark('Wikitext-accuracy')
        benchmark.data = benchmark.data[0:2]  # test on subset for speed
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
        score = benchmark(model)
        assert score == 0


class TestReadingTimes:
    def test_single_word(self):
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
        model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_time = model.digest_text('the')['behavior']
        assert np.isnan(reading_time)

    def test_multiple_words(self):
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
        model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_time = model.digest_text('the quick brown fox')['behavior']
        assert reading_time == approx(44.06524, abs=0.001)

    def test_list_input(self):
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
        text = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy']
        _logger.info(f'Running {model.identifier()} with text "{text}"')
        model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_times = model.digest_text(text)['behavior']
        np.testing.assert_allclose(
            reading_times, [np.nan, 19.260605, 12.721411, 12.083241, 10.876629, 3.678278, 2.102749, 11.961533],
            atol=0.0001)

    def test_multitoken_words(self):
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
        text = ['beekeepers', 'often', 'go', 'beekeeping']
        _logger.info(f'Running {model.identifier()} with text "{text}"')
        model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_times = model.digest_text(text)['behavior']
        np.testing.assert_allclose(
            reading_times, [16.1442, 10.4003, 6.6620, 16.0906 + 1.3748], atol=0.0001)

    def test_multiword_list_input(self):
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
        text = ['the quick brown fox', 'jumps over', 'the lazy']
        _logger.info(f'Running {model.identifier()} with text "{text}"')
        model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_times = model.digest_text(text)['behavior']
        np.testing.assert_allclose(reading_times, [44.06524, 14.554907, 14.064276], atol=0.0001)

    def test_punct(self):
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={})
        text = ['fox', 'is', 'quick.']
        _logger.info(f'Running {model.identifier()} with text "{text}"')
        model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_times = model.digest_text(text)['behavior']
        np.testing.assert_allclose(
            reading_times, [np.nan, 8.422014, 11.861147 + 5.9755263], atol=0.0001)

    @pytest.mark.memory_intense
    def test_tokenizer_eos(self):
        """
        Test model outputs for a model whose tokenizer inserts EOS/BOS tokens.
        """
        model = HuggingfaceSubject(model_id='xlm-roberta-base', region_layer_mapping={})
        text = ['the quick brown fox', 'jumps over', 'the lazy dog']
        # expected tokenization:
        # ['<s>', '▁The', '▁quick', '▁brown', '▁', 'fox', '▁jump', 's', '▁over', '▁the',
        #  '▁la', 'zy', '▁dog', '</s>']
        _logger.info(f'Running {model.identifier()} with text "{text}"')
        model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_times = model.digest_text(text)['behavior']
        np.testing.assert_allclose(
            reading_times, [139.66634, 111.14671, 136.64108], atol=0.0001)


class TestNeural:
    def test_list_input(self):
        """
        This is a simple test that takes in text = ['the quick brown fox', 'jumps over', 'the lazy'], and tests the
        representation for next word prediction for each sentence in the list.
        This test is a stand-in prototype to check if our model definitions are correct.
        """

        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system: 'transformer.h.0.ln_1'})
        text = ['the quick brown fox', 'jumps over', 'the lazy dog']
        _logger.info(f'Running {model.identifier()} with text "{text}"')
        model.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                     recording_type=ArtificialSubject.RecordingType.fMRI)
        representations = model.digest_text(text)['neural']
        assert len(representations['presentation']) == 3
        np.testing.assert_array_equal(representations['stimulus'], text)
        assert len(representations['neuroid']) == 768

    @pytest.mark.memory_intense
    def test_one_text_single_target(self):
        """
        This is a simple test that takes in text = 'the quick brown fox', and asserts that the `distilgpt2` layer
        indexed by `representation_layer` has 1 text presentation and 768 neurons. This test is a stand-in prototype to
        check if our model definitions are correct.
        """
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system: 'transformer.h.0.ln_1'})
        text = 'the quick brown fox'
        _logger.info(f'Running {model.identifier()} with text "{text}"')
        model.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                     recording_type=ArtificialSubject.RecordingType.fMRI)
        representations = model.digest_text(text)['neural']
        assert len(representations['presentation']) == 1
        assert representations['stimulus'].squeeze() == text
        assert len(representations['neuroid']) == 768
        _logger.info(f'representation shape is correct: {representations.shape}')

    @pytest.mark.memory_intense
    def test_one_text_single_target_bidirectional(self):
        """
        This is a simple test that takes in text = 'the quick brown fox', and asserts that a bidirectiona BERT model
        layer indexed by `representation_layer` has 1 text presentation and 768 neurons. This test is a stand-in prototype
        to check if our model definitions are correct.
        """
        model = HuggingfaceSubject(model_id='bert-base-uncased', region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system: 'bert.encoder.layer.4'})
        text = 'the quick brown fox'
        _logger.info(f'Running {model.identifier()} with text "{text}"')
        model.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                     recording_type=ArtificialSubject.RecordingType.fMRI)
        representations = model.digest_text(text)['neural']
        assert len(representations['presentation']) == 1
        assert representations['stimulus'].squeeze() == text
        assert len(representations['neuroid']) == 768
        _logger.info(f'representation shape is correct: {representations.shape}')

    @pytest.mark.memory_intense
    def test_one_text_two_targets(self):
        model = HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system_left_hemisphere: 'transformer.h.0.ln_1',
            ArtificialSubject.RecordingTarget.language_system_right_hemisphere: 'transformer.h.1.ln_1'})
        text = 'the quick brown fox'
        _logger.info(f'Running {model.identifier()} with text "{text}"')
        model.start_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system_left_hemisphere,
            recording_type=ArtificialSubject.RecordingType.fMRI)
        model.start_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system_right_hemisphere,
            recording_type=ArtificialSubject.RecordingType.fMRI)
        representations = model.digest_text(text)['neural']
        assert len(representations['presentation']) == 1
        assert representations['stimulus'].squeeze() == text
        assert len(representations['neuroid']) == 768 * 2
        assert set(representations['region'].values) == {
            ArtificialSubject.RecordingTarget.language_system_left_hemisphere,
            ArtificialSubject.RecordingTarget.language_system_right_hemisphere}
        _logger.info(f'representation shape is correct: {representations.shape}')
