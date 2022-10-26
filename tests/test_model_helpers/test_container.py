import logging

import numpy as np
import pytest
from pytest import approx

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.model_helpers.container import ContainerSubject

logging.basicConfig(level=logging.INFO)


class TestNextWord:
    @pytest.mark.parametrize(
        "model_identifier, expected_next_word",
        [
            pytest.param("rnn-lm-ptb", "in", marks=pytest.mark.memory_intense),
        ],
    )
    def test_single_string(self, model_identifier, expected_next_word):
        """
        This is a simple test that takes in text = 'the quick brown fox', and tests the next word.
        This test is a stand-in prototype to check if our model definitions are correct.
        """

        model = load_model(model_identifier)
        text = "the quick brown fox jumps"
        logging.info(f'Running {model_identifier} with text "{text}"')
        model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
        next_word = model.digest_text(text)["behavior"].values
        assert next_word == expected_next_word

    @pytest.mark.parametrize(
        "model_identifier, expected_next_words",
        [
            pytest.param(
                "rnn-lm-ptb", ["in", "the", "of"], marks=pytest.mark.memory_intense
            ),
        ],
    )
    def test_list_input(self, model_identifier, expected_next_words):
        """
        This is a simple test that takes in a list of text parts,
        and tests the next word for each text part in the list.
        This test is a stand-in prototype to check if our model definitions are correct.
        """
        model = load_model(model_identifier)
        text = ["the quick brown fox jumps", "over", "the lazy dog"]
        logging.info(f'Running {model_identifier} with text "{text}"')
        model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
        next_words = model.digest_text(text)["behavior"]
        np.testing.assert_array_equal(next_words, expected_next_words)


class TestReadingTimes:
    def test_single_word(self):
        model = load_model("rnn-lm-ptb")
        model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_time = model.digest_text("the")["behavior"]
        assert not np.isnan(reading_time)  # predicts first token from <s> token

    def test_multiple_words(self):
        model = load_model("rnn-lm-ptb")
        model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_time = model.digest_text("the quick brown fox")["behavior"]
        assert reading_time == approx(46.92789, abs=0.01)

    def test_list_input(self):
        model = load_model("rnn-lm-ptb")
        text = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy"]
        model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_times = model.digest_text(text)["behavior"]
        np.testing.assert_allclose(
            reading_times,
            [10.370, 15.490, 18.165, 2.902, 15.848, 6.3294, 1.238, 15.902],
            atol=0.01,
        )

    def test_multitoken_words(self):
        # (other) will be split to 3 tokens ["-LRB-","other","-RRB-"]
        model = load_model("rnn-lm-ptb")
        text = ["I", "saw", "the", "(other)", "dog"]
        model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_times = model.digest_text(text)["behavior"]
        np.testing.assert_allclose(
            reading_times, [7.395, 7.759, 2.702, 27.936, 13.640], atol=0.01
        )

    def test_multiword_list_input(self):
        model = load_model("rnn-lm-ptb")
        text = ["the quick brown fox", "jumps over", "the lazy"]
        model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_times = model.digest_text(text)["behavior"]
        np.testing.assert_allclose(reading_times, [46.928, 22.178, 17.140], atol=0.01)

    def test_punct(self):
        model = load_model("rnn-lm-ptb")
        text = 'The fox, "Brian", is quick. (too quick?)'
        model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
        reading_times = model.digest_text(text)["behavior"]
        np.testing.assert_allclose(reading_times, [81.160], atol=0.01)


class TestNeural:
    def test_list_input(self):
        """
        This is a simple test that takes in text = ['the quick brown fox', 'jumps over', 'the lazy'], and tests the
        representation for next word prediction for each sentence in the list.
        This test is a stand-in prototype to check if our model definitions are correct.
        """
        model = load_model("rnn-lm-ptb")
        text = ["the quick brown fox", "jumps over", "the lazy dog"]
        model.start_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system,
            recording_type=ArtificialSubject.RecordingType.fMRI,
        )
        representations = model.digest_text(text)["neural"]
        assert len(representations["presentation"]) == 3
        np.testing.assert_array_equal(representations["stimulus"], text)
        assert len(representations["neuroid"]) == 650

    @pytest.mark.memory_intense
    def test_one_text_single_target(self):
        """
        This is a simple test that takes in text = 'the quick brown fox', and asserts that the `rnn-lm-ptb` representation
        indexed by `representation_layer` has 1 text presentation and 768 neurons. This test is a stand-in prototype to
        check if our model definitions are correct.
        """
        model = load_model("rnn-lm-ptb")
        text = "the quick brown fox"
        model.start_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system,
            recording_type=ArtificialSubject.RecordingType.fMRI,
        )
        representations = model.digest_text(text)["neural"]
        assert len(representations["presentation"]) == 1
        assert representations["stimulus"].squeeze() == text
        assert len(representations["neuroid"]) == 650

    @pytest.mark.memory_intense
    def test_one_text_two_targets(self):
        model = ContainerSubject(
            container="benlipkin/rnng:6f6825d1c4a8c58c844b4b82123b967bb0bab6ce",
            entrypoint="cd /app && source activate rnng && python -m brainscore",
            identifier="rnn-lm-ptb",
            region_layer_mapping={
                ArtificialSubject.RecordingTarget.language_system_left_hemisphere: "lstm-mean",
                ArtificialSubject.RecordingTarget.language_system_right_hemisphere: "emb-mean",
            },
        )
        text = "the quick brown fox"
        model.start_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system_left_hemisphere,
            recording_type=ArtificialSubject.RecordingType.fMRI,
        )
        model.start_neural_recording(
            recording_target=ArtificialSubject.RecordingTarget.language_system_right_hemisphere,
            recording_type=ArtificialSubject.RecordingType.fMRI,
        )
        representations = model.digest_text(text)["neural"]
        assert len(representations["presentation"]) == 1
        assert representations["stimulus"].squeeze() == text
        assert len(representations["neuroid"]) == 650 * 2
        assert set(representations["region"].values) == {
            ArtificialSubject.RecordingTarget.language_system_left_hemisphere,
            ArtificialSubject.RecordingTarget.language_system_right_hemisphere,
        }
