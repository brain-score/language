"""
Defines methods for aligning LM tokens with suite regions, and converting
raw token-level surprisal outputs from models into suites with computed
region-level surprisals.
"""
import re
import sys
import warnings

import numpy as np
import pandas as pd

from copy import deepcopy
from typing import List, Tuple, NamedTuple, Mapping

from brainscore_language.plugins.sg_tse.suite import Suite, Region

METRICS = {
    'sum': sum,
    'mean': np.mean,
    'median': np.median,
    'range': np.ptp,
    'max': max,
    'min': min
}

def validate_metrics(metrics):
    """
    Checks if specified metrics are valid. Returns None if check passes,
    else raises ValueError.
    """
    if any(m not in METRICS for m in metrics):
        bad_metrics = [m for m in metrics if m not in METRICS]
        raise ValueError('Unknown metrics: {}'.format(bad_metrics))
def _prepare_metrics(suite: Suite) -> List[str]:
    # check that specified metrics are implemented in utils.METRICS
    metrics = suite.meta["metric"]
    if metrics == 'all':
        metrics = METRICS.keys()
    else:
        # if only one metric specified, convert to singleton list
        metrics = [metrics] if type(metrics) == str else metrics
    validate_metrics(metrics)

    return metrics


class ItemSentenceMapping(NamedTuple):
    """
    Represents a mapping between a tokenized sentence output from a model and
    a sentence as encoded in a test suite (i.e. an item--condition pair).
    Used to map token-level model surprisals to region-level test suite
    surprisals.
    """
    id: Tuple[int, str]
    """
    (item_number, condition_name)
    """

    region_to_tokens: Mapping[int, List[str]]
    """
    For each region of the item--condition, maps to a list of corresponding
    tokens in the model output.
    """

    oovs: Mapping[int, List[str]]
    """
    For each region of the item--condition, maps to a list of spans in the
    region content which were marked as out-of-vocabulary by the model.
    """

def prepare_sentences_huggingface(model, tokens: List[List[str]],
                                  suite: Suite) -> List[ItemSentenceMapping]:
    """
    Compute token-to-region mapping for each sentence in the suite. This
    implementation uses Huggingface models' detokenization information and
    should be more robust than the heuristic method.
    """

    region_edges = list(suite.iter_region_edges())

    # Hack: re-tokenize here in order to detokenize back to character-level
    # offsets.
    sentences = list(suite.iter_sentences())
    encoded = model.tokenizer.batch_encode_plus(
        sentences, add_special_tokens=True, return_offsets_mapping=True)

    ret: List[ItemSentenceMapping] = []

    sent_idx = 0
    for i_idx, item in enumerate(suite.items):
        for c_idx, cond in enumerate(item["conditions"]):
            regions = [Region(**r) for r in cond["regions"]]

            ret.append(compute_mapping_huggingface(
                encoded.tokens(sent_idx),
                regions,
                encoded["offset_mapping"][sent_idx],
                region_edges[sent_idx],
                item_number=i_idx + 1,
                condition_name=cond["condition_name"]
            ))

            sent_idx += 1

    return ret

def compute_mapping_huggingface(tokens: List[str], regions: List[Region],
                                token_offsets: List[Tuple[int, int]],
                                region_edges: List[int],
                                item_number=None,
                                condition_name=None) -> ItemSentenceMapping:
    region2tokens = {r.region_number: [] for r in regions}
    r_cursor, t_cursor = 0, 0
    while t_cursor < len(tokens):
        token = tokens[t_cursor]
        token_char_start, token_char_end = token_offsets[t_cursor]

        region_start = region_edges[r_cursor]
        region_end = region_edges[r_cursor + 1] \
            if r_cursor + 1 < len(region_edges) else np.inf

        # NB region boundaries are left edges, hence the >= here.
        if token_char_start >= region_end:
            r_cursor += 1
            continue

        region2tokens[r_cursor + 1].append(token)
        t_cursor += 1

    return ItemSentenceMapping(
        id=(item_number, condition_name),
        region_to_tokens=region2tokens,
        oovs={region: []
              for region in region2tokens.keys()})


def aggregate_surprisals(model, surprisals: pd.DataFrame,
                         tokens: List[List[str]], suite: Suite):
    metrics = _prepare_metrics(suite)

    ret = deepcopy(suite)
    surprisals = surprisals.reset_index().set_index("sentence_id")

    # Checks
    sent_idx = 0
    for i_idx, item in enumerate(suite.items):
        for c_idx, cond in enumerate(item['conditions']):
            # fetch sentence data
            sent_tokens = tokens[sent_idx]
            sent_surps = surprisals.loc[sent_idx + 1]

            if sent_tokens != list(sent_surps.token):
                raise ValueError("Mismatched tokens between tokens and surprisals data frame")

    # Run sentence prep procedure -- map tokens in each sentence onto regions
    # of corresponding test trial sentence

        mapper = prepare_sentences_huggingface


    sentence_mappings: List[ItemSentenceMapping] = mapper(model, tokens, suite)

    # Bring in surprisals
    sent_idx = 0
    for i_idx, item in enumerate(suite.items):
        for c_idx, cond in enumerate(item["conditions"]):
            sent_mapping = sentence_mappings[sent_idx]

            sent_tokens = tokens[sent_idx]
            sent_surps = surprisals.loc[sent_idx + 1].surprisal.values

            # iterate through regions in sentence
            t_idx = 0
            for r_idx, (region_number, region_tokens) in enumerate(sent_mapping.region_to_tokens.items()):
                region_surprisals = []
                for token in region_tokens:
                    # append to region surprisals if exact token match
                    if token == sent_tokens[t_idx]:
                        region_surprisals.append(sent_surps[t_idx])
                        t_idx += 1
                    else:
                        raise utils.TokenMismatch(token, sent_tokens[t_idx], t_idx+2)

                # get dictionary of region-level surprisal values for each metric
                vals = {m: METRICS[m](region_surprisals)
                        for m in metrics}

                # insert surprisal values into original dict
                ret.items[i_idx]['conditions'][c_idx]['regions'][r_idx]['metric_value'] = vals

                # update original dict with OOV information
                ret.items[i_idx]['conditions'][c_idx]['regions'][r_idx]['oovs'] = \
                  sent_mapping.oovs[region_number]

            # update sentence counter
            sent_idx += 1

    return ret
