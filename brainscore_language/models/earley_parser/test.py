from pathlib import Path
import numpy as np
import pytest

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject


class grammars:
    GRAMMAR_1 = """
                    S -> NP VP [1.0]
                    NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]
                    Det -> 'the' [0.8] | 'my' [0.2]
                    N -> 'man' [0.5] | 'telescope' [0.5]
                    VP -> VP PP [0.1] | V NP [0.7] | V [0.2]
                    V -> 'ate' [0.35] | 'saw' [0.65]
                    PP -> P NP [1.0]
                    P -> 'with' [0.61] | 'under' [0.39]
                """

    GRAMMAR_2 = """
                S    -> NP VP         [1.0]
                VP   -> V NP          [.59]
                VP   -> V             [.40]
                VP   -> VP PP         [.01]
                NP   -> Det N         [.41]
                NP   -> Name          [.28]
                NP   -> NP PP         [.31]
                PP   -> P NP          [1.0]
                V    -> 'saw'         [.21]
                V    -> 'ate'         [.51]
                V    -> 'ran'         [.28]
                N    -> 'boy'         [.11]
                N    -> 'cookie'      [.12]
                N    -> 'table'       [.13]
                N    -> 'telescope'   [.14]
                N    -> 'hill'        [.5]
                Name -> 'Jack'        [.52]
                Name -> 'Bob'         [.48]
                P    -> 'with'        [.61]
                P    -> 'under'       [.39]
                Det  -> 'the'         [.41]
                Det  -> 'a'           [.31]
                Det  -> 'my'          [.28]
            """


@pytest.mark.parametrize(
    "model_identifier, treebank_path, expected_reading_times, expected_next_words",
    [
        (
            "earley-parser",
            str(Path(__file__).parent / "treebank"),
            [1.15200, 3.4739, 6.2109, 8.08537, 9.4073, 10.5593, 12.5593, 13.8812],
            ["lazy", "man", "saw", "with", "a", "lazy", "man", "<unk>"],
        ),
    ],
)
def test_create_grammar(
    model_identifier, treebank_path, expected_reading_times, expected_next_words
):
    model = load_model(model_identifier)
    model.create_grammar(
        treebank_path=treebank_path,
        fileids="sample_treebank",
    )
    text = ["the", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
    model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
    reading_times = model.digest_text(text)["behavior"]
    np.testing.assert_allclose(reading_times, expected_reading_times, atol=0.0001)

    model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
    next_words = model.digest_text(text)["behavior"]
    np.testing.assert_array_equal(next_words, expected_next_words)


@pytest.mark.parametrize(
    "model_identifier, grammar_string, expected_reading_times",
    [
        (
            "earley-parser",
            grammars.GRAMMAR_1,
            [2.4150, 3.1885, 6.4709, 7.1840, 10.5060, 11.5060],
        ),
    ],
)
def test_reading_times_1(model_identifier, grammar_string, expected_reading_times):
    model = load_model(model_identifier)
    model.set_grammar(grammar_string)
    text = ["I", "saw", "John", "with", "my", "telescope"]
    model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
    reading_times = model.digest_text(text)["behavior"]
    np.testing.assert_allclose(reading_times, expected_reading_times, atol=0.0001)


@pytest.mark.parametrize(
    "model_identifier, grammar_string, expected_reading_times",
    [
        (
            "earley-parser",
            grammars.GRAMMAR_2,
            [2.7799, 5.0460, 7.9414, 8.6545, 11.7773, 14.6138],
        ),
    ],
)
def test_reading_times_2(model_identifier, grammar_string, expected_reading_times):
    model = load_model(model_identifier)
    model.set_grammar(grammar_string)
    text = ["Jack", "saw", "Bob", "with", "my", "telescope"]
    model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
    reading_times = model.digest_text(text)["behavior"]
    np.testing.assert_allclose(reading_times, expected_reading_times, atol=0.0001)


@pytest.mark.parametrize(
    "model_identifier, grammar_string, expected_next_words",
    [
        (
            "earley-parser",
            grammars.GRAMMAR_1,
            ["saw", "the", "with", "the", "man", "with"],
        ),
    ],
)
def test_next_word_1(model_identifier, grammar_string, expected_next_words):
    model = load_model(model_identifier)
    model.set_grammar(grammar_string)
    text = ["I", "saw", "John", "with", "my", "telescope"]
    model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
    next_words = model.digest_text(text)["behavior"]
    np.testing.assert_array_equal(next_words, expected_next_words)


@pytest.mark.parametrize(
    "model_identifier, grammar_string, expected_next_words",
    [
        (
            "earley-parser",
            grammars.GRAMMAR_2,
            ["ate", "the", "with", "the", "hill", "with"],
        ),
    ],
)
def test_next_word_2(model_identifier, grammar_string, expected_next_words):
    model = load_model(model_identifier)
    model.set_grammar(grammar_string)
    text = ["Jack", "saw", "Bob", "with", "my", "telescope"]
    model.start_behavioral_task(task=ArtificialSubject.Task.next_word)
    next_words = model.digest_text(text)["behavior"]
    np.testing.assert_array_equal(next_words, expected_next_words)
