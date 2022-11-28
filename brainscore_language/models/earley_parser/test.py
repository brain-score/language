import numpy as np
import pytest

from brainscore_language import load_model
from brainscore_language.artificial_subject import ArtificialSubject


@pytest.mark.parametrize(
    "model_identifier, grammar_string, expected_reading_times",
    [
        (
            "earley-parser",
            """
                S -> NP VP [1.0]
                NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]
                Det -> 'the' [0.8] | 'my' [0.2]
                N -> 'man' [0.5] | 'telescope' [0.5]
                VP -> VP PP [0.1] | V NP [0.7] | V [0.2]
                V -> 'ate' [0.35] | 'saw' [0.65]
                PP -> P NP [1.0]
                P -> 'with' [0.61] | 'under' [0.39]
            """,
            [7.1949, 5.035],  # TODO
        ),
    ],
)
def test_reading_times_1(model_identifier, grammar_string, expected_reading_times):
    model = load_model(model_identifier)
    model.set_grammar(grammar_string)
    text = ["I saw John", "with my telescope"]
    model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
    reading_times = model.digest_text(text)["behavior"]
    np.testing.assert_allclose(reading_times, expected_reading_times, atol=0.0001)


@pytest.mark.parametrize(
    "model_identifier, grammar_string, expected_reading_times",
    [
        (
            "earley-parser",
            """
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
            """,
            [8.688, 6.6724],  # TODO
        ),
    ],
)
def test_reading_times_2(model_identifier, grammar_string, expected_reading_times):
    model = load_model(model_identifier)
    model.set_grammar(grammar_string)
    text = ["Jack saw Bob", "with my telescope"]
    model.start_behavioral_task(task=ArtificialSubject.Task.reading_times)
    reading_times = model.digest_text(text)["behavior"]
    np.testing.assert_allclose(reading_times, expected_reading_times, atol=0.0001)