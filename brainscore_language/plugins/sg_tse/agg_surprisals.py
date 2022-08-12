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


def prepare_sentences(model, tokens: List[List[str]],
                      suite: Suite) -> List[ItemSentenceMapping]:
    """
    Compute token-to-region mapping for each sentence in the suite. This is the
    default heuristic implementation.
    """
    sent_idx = 0
    ret = []

    # Pre-fetch model spec for aggregation algorithm
    model_spec = spec(model)

    for i_idx, item in enumerate(suite.items):
        for c_idx, cond in enumerate(item['conditions']):
            sent_tokens = tokens[sent_idx]
            regions = [Region(**r) for r in cond["regions"]]

            try:
                mapping = compute_mapping_heuristic(
                    sent_tokens, regions, model_spec,
                    item_number=i_idx + 1,
                    condition_name=cond["condition_name"])
            except Exception as e:
                print("Tokens: ", sent_tokens, file=sys.stderr)
                print("Region spec: ", cond["regions"], file=sys.stderr)

                raise ValueError("Error occurred while processing item %i, "
                                 "condition %s. Relevant debug information "
                                 "printed to stderr."
                                 % (item["item_number"],
                                    cond["condition_name"])) from e

            ret.append(mapping)
            sent_idx += 1

    return ret


MOSES_PUNCT_SPLIT_TOKEN = re.compile(r"^@([-,.])@$")
"""
Moses tokenizers split intra-token hyphens and decimal separators , and .
into separate tokens, using @ as a sentinel for detokenization. We account for
this when detokenizing here.
"""
def compute_mapping_heuristic(tokens: List[str], regions: List[Region],
                              model_spec: dict,
                              item_number=None,
                              condition_name=None) -> ItemSentenceMapping:
    def get_next_region(r_idx):
        r = regions[r_idx]
        return r, r.content

    # initialize variables
    r_idx = 0
    r, content = get_next_region(r_idx)

    region2tokens = {region.region_number: [] for region in regions}
    oovs: Mapping[int, List[str]] = {region.region_number: []
                                     for region in regions}

    # compile regex for dropping
    if model_spec['tokenizer'].get('drop_token_pattern') is not None:
        drop_pattern = re.compile(model_spec['tokenizer']['drop_token_pattern'])
    else:
        drop_pattern = None
    metaspace = model_spec["tokenizer"].get("metaspace")

    # Sentinel: blindly add next N tokens to current region.
    skip_n = 0

    # iterate over all tokens in sentence
    t_idx = 0
    while t_idx < len(tokens):
        token = tokens[t_idx]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Token-level operations
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # append and continue upon encountering start-of-sentence token
        if token in model_spec['vocabulary']['prefix_types']:
            region2tokens[r.region_number].append(token)
            t_idx += 1
            continue

        # exit loop upon encountering end-of-sentence token
        elif token in model_spec['vocabulary']['suffix_types']:
            region2tokens[r.region_number].append(token)
            break

        # skip current token for special cases
        elif token in model_spec['vocabulary']['special_types']:
            # TODO: which region should special_type associate with?
            t_idx += 1
            continue

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Content-level operations
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # remove leading spaces of current content
        content = content.lstrip()

        # drop characters specified by regex (content up to next space)
        if (content != '' and drop_pattern
            and re.sub(drop_pattern, '', content.split()[0]) == ''):
            content = ' '.join(content.split()[1:])

        # if empty region, proceed to next region (keeping current token)
        if content == '':
            r_idx += 1
            r, content = get_next_region(r_idx)
            continue

        # remove casing if necessary
        if not model_spec['tokenizer']['cased']:
            content = content.lower()

        # Check for a token match at the left edge of the region.
        step_count = None
        token_match = content.startswith(token)
        if token_match:
            # Exact match. We'll walk forward this number of characters
            step_count = len(token)
        # Subword tokenizers may have initial / final content that blocks
        # the match. Check again.
        if not token_match and model_spec["tokenizer"]["type"] == "subword":
            if model_spec["tokenizer"]["sentinel_position"] in ["initial", "medial"]:
                stripped_token = token.lstrip(model_spec["tokenizer"]["sentinel_pattern"])
            elif model_spec["tokenizer"]["sentinel_position"] == "final":
                stripped_token = token.rstrip(model_spec["tokenizer"]["sentinel_pattern"])

            token_match = content.startswith(stripped_token)
            # Soft subword match. Step forward the number of characters in
            # the matched subword, correcting for sentinel
            if token_match:
                step_count = len(stripped_token)

        if not token_match and metaspace is not None and token.startswith(metaspace):
            token_match = True

            # metaspace may end up as its own token or joined with
            # surrounding content. account for both cases.
            if len(token) > len(metaspace):
                step_count = len(token) - len(metaspace)
            else:
                # if metaspace was on its own, we don't need to advance the
                # reference string -- the corresponding space was already
                # stripped by lstrip() call above.
                step_count = 0

        # Account for Moses sentinel if relevant.
        if "moses" in model_spec["tokenizer"].get("behaviors", []) \
            and MOSES_PUNCT_SPLIT_TOKEN.match(token):
            # Match. Step forward the number of characters between the Moses
            # @ sentinel.
            token_match = True
            stripped_token = MOSES_PUNCT_SPLIT_TOKEN.match(token).group(1)
            step_count = len(stripped_token)

        # If we found a left-edge match, or this is an unk
        if token_match or token in model_spec['vocabulary']['unk_types']:

            # First: consume the (soft) matched token.
            if token_match:
                # add token to list of tokens for current region
                region2tokens[r.region_number].append(token)

                # remove token from content
                content = content[step_count:]
                t_idx += 1
            else:
                # extract maximal string of OOVs by looking for match with
                # next non-OOV token
                tokens_remaining = len(tokens) - t_idx
                oov_str = None
                for token_window_size in range(1, tokens_remaining+1):
                    # token_window_size is number of tokens to look ahead
                    if token_window_size == tokens_remaining:
                        # No fancy work needed here -- we're consuming the
                        # entire remainder of the string. Add to current
                        # region and quit.
                        region2tokens[r.region_number].extend(tokens[t_idx:])

                        oov_str = " ".join([content] + [r.content for r in regions[r_idx + 1:]])
                        oovs[r.region_number].extend(oov_str.split())

                        t_idx += token_window_size

                        break
                    else:
                        if token_window_size > 1:
                            warnings.warn(
                                (f'Consecutive OOVs found in '
                                 f'Item {item_number}, '
                                 f'Condition "{condition_name}"!'),
                                RuntimeWarning)

                        next_token = tokens[t_idx + token_window_size]
                        next_token_is_punct = re.match(r"\W+", next_token)

                        # Eat up content across regions until we come to a
                        # token that we can match with `next_token`.
                        eaten_content = []
                        for next_r_idx in range(r_idx, len(regions)):
                            if oov_str:
                                # OOV resolution is complete. Break.
                                break

                            if next_r_idx == r_idx:
                                next_r_content = content
                            else:
                                eaten_content.append(next_r_content.strip())
                                _, next_r_content = get_next_region(next_r_idx)

                            for i in range(len(next_r_content)):
                                # When searching for a word-like token
                                # (not punctuation), only allow matches at
                                # token boundaries in region content.
                                # This protects against the edge case where
                                # a substring of the unk'ed token matches
                                # a succeeding content in the token, e.g.
                                #   content: "will remand and order"
                                #   tokens: "will <unk> and order"
                                #
                                # See test case "remand test"
                                if not next_token_is_punct and \
                                  (i > 0 and not re.match(r"\W", next_r_content[i - 1])):
                                    continue

                                if next_r_content[i:i+len(next_token)] == next_token:
                                    # We found a token which faithfully
                                    # matches the reference token. Break
                                    # just before that token.
                                    eaten_content.append(next_r_content[:i].strip())
                                    # NB, we use `oov_str` as a sentinel
                                    # marking that the match is complete
                                    oov_str = " ".join(eaten_content).strip()

                                    # track OOVs -- put them in the
                                    # leftmost associated region
                                    oovs[r.region_number].extend(oov_str.split(" "))

                                    # Blindly add all these eaten tokens
                                    # from the content to the leftmost
                                    # region -- not including the token
                                    # that just matched, of course.
                                    region2tokens[r.region_number].extend(
                                        tokens[t_idx:t_idx + token_window_size])
                                    t_idx += token_window_size

                                    # Update the current region reference.
                                    r_idx = next_r_idx
                                    r = regions[r_idx]
                                    content = next_r_content[i:]

                                    break

                        if oov_str:
                            # OOV resolution is complete. Break.
                            break

                if content.strip() == '' and r_idx == len(regions) - 1:
                    # TODO break
                    return region2tokens

            # if end of content (removing spaces), and before last region
            if content.strip() == '' and r_idx < len(regions) - 1:
                # warn user
                if r_idx > 0 and oovs[r_idx]:
                    warnings.warn(
                        (f'OOVs found in Item {item_number}, '
                         f'Condition "{condition_name}": "{oovs[r_idx]}"'),
                        RuntimeWarning)
                r_idx += 1
                r, content = get_next_region(r_idx)

        # otherwise, move to next region and token
        else:
            t_idx += 1
            r_idx += 1
            r, content = get_next_region(r_idx)

    return ItemSentenceMapping(
        id=(item_number, condition_name),
        region_to_tokens=region2tokens,
        oovs=oovs
    )


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

    # update meta information with model name
  #  ret.meta['model'] = spec(model)['name']
    return ret
