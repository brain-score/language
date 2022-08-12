import json
import torch

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Union, Dict, TextIO, List

from brainscore_language.models.huggingface import HuggingfaceSubject
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.plugins.sg_tse.agg_surprisals import aggregate_surprisals
from brainscore_language.plugins.sg_tse.suite import Suite

def _load_suite(suite_ref: Union[str, Path, TextIO, Dict, Suite]) -> Suite:
    pass
    if isinstance(suite_ref, Suite):
         return suite_ref

    # Load from dict / JSON file / JSON path
    if not isinstance(suite_ref, dict):
         if not hasattr(suite_ref, "read"):
             suite_ref = open(suite_ref, "r")
         suite = json.load(suite_ref)
    else:
         suite = suite_ref
    return Suite.from_dict(suite)


def compute_surprisals(model: ArtificialSubject, suite) -> Suite:
    """
    Compute per-region surprisals for a language model on the given suite.

    Args:
        model: A Brain-Score Language ``Model`` following the ArtificialSubject API.
        suite_file: A path or open file stream to a suite JSON file, or an
            already loaded suite dict

    Returns:
        An evaluated test suite dict --- a copy of the data from
        ``suite_file``, now including per-region surprisal data
    """
    suite = _load_suite(suite)

    # Convert to sentences
    suite_sentences = list(suite.iter_sentences())

    # First compute surprisals
    surprisals_df = get_surprisals(model, suite_sentences)

    # Track tokens
    tokens = tokenize(model, suite_sentences)

    # Now aggregate over regions and get result df
    result = aggregate_surprisals(model, surprisals_df, tokens, suite)

    return result


def evaluate(suite: Suite, return_df=True):
    """
    Evaluate prediction results on the given suite. The suite must contain
    surprisal estimates for all regions.
    """
    suite = _load_suite(suite)
    results = suite.evaluate_predictions()
    if not return_df:
        return suite, results

    # Make a nice dataframe
    results_data = [(suite.meta["name"], pred.idx, item_number, result)
                    for item_number, preds in results.items()
                    for pred, result in preds.items()]
    return pd.DataFrame(results_data, columns=["suite", "prediction_id", "item_number", "result"]) \
            .set_index(["suite", "prediction_id", "item_number"])


def _get_predictions_inner(model: HuggingfaceSubject, sentence: str):
    sent_tokens = model.tokenizer.tokenize(sentence, add_special_tokens=True)
    indexed_tokens = model.tokenizer.convert_tokens_to_ids(sent_tokens)
    # create 1 * T input token tensor
    tokens_tensor = torch.tensor(indexed_tokens).unsqueeze(0)

    with torch.no_grad():
        log_probs = model.basemodel(tokens_tensor)[0] \
            .log_softmax(dim=2).squeeze()

    return list(zip(sent_tokens, indexed_tokens,
                    (None,) + log_probs.unbind()))

def tokenize(model: HuggingfaceSubject, sentences: List[str]) -> List[List[str]]:
    return [model.tokenizer.tokenize(sentence, add_special_tokens=True)
            for sentence in sentences]

def get_surprisals(model: HuggingfaceSubject, sentences: List[str]) -> pd.DataFrame:
    df = []
    columns = ["sentence_id", "token_id", "token", "surprisal"]
    for i, sentence in enumerate(sentences):
        predictions = _get_predictions_inner(model, sentence)

        for j, (word, word_idx, preds) in enumerate(predictions):
            if preds is None:
                surprisal = 0.0
            else:
                surprisal = -preds[word_idx].item() / np.log(2)

            df.append((i + 1, j + 1, word, surprisal))

    return pd.DataFrame(df, columns=columns) \
        .set_index(["sentence_id", "token_id"])


