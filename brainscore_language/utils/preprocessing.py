import re


_space_before_punctuation_re = re.compile(r'\s+([.,!?;:])')

def prepare_context(context_parts: list[str]) -> str:
    """
    Prepare a context for use in a neural or behavioral task. Joins
    the given list of natural-language context part strings and adjusts
    for any resulting artifacts.

    Note that this implementation is English-specific.
    """

    # Drop empty parts -- otherwise we will create double-spaces in the result.
    context_parts = [part for part in context_parts if part.strip() != ""]

    context = " ".join(context_parts)

    # Remove erroneous spaces before punctuation.
    context = _space_before_punctuation_re.sub(r'\1', context)

    return context