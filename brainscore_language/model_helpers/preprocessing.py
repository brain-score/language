from typing import List
from brainscore_core.text import prepare_context as _prepare_context


def prepare_context(context_parts: List[str]) -> str:
    """
    Prepare a context for use in a neural or behavioral task. Joins
    the given list of natural-language context part strings and adjusts
    for any resulting artifacts.

    Note that this implementation is English-specific.
    """

    return _prepare_context(context_parts)
