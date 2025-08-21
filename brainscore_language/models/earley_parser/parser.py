import logging
from pathlib import Path
from typing import Tuple, Union, List, Dict, Optional

import numpy as np
import xarray as xr
import nltk
from nltk.grammar import PCFG, Nonterminal, is_terminal
from nltk.corpus.reader import BracketParseCorpusReader
from nltk.parse.pchart import Chart

from brainio.assemblies import DataAssembly, BehavioralAssembly
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils import fullname

from brainscore_language.models.earley_parser.utils import (
    ProbabilisticEarleyChartParser,
)

PARSING_TRACE = 0  # how verbose the tracing output should be while parsing a text


class EarleyParserSubject(ArtificialSubject):
    def __init__(self):
        self._logger = logging.getLogger(fullname(self))
        self.model_id = "earley-parser-minivocab"
        self.parser_cls = ProbabilisticEarleyChartParser

        self.grammar: Optional[PCFG] = None

        # Load the default grammar. This can be replaced by any custom treebank
        treebank_path = str(Path(__file__).parent / "treebank")
        self.create_grammar(
            treebank_path=treebank_path,
            fileids="sample_treebank",
        )

        self.prefix_probabilities = []  # Records the partial parse probabilities

        self.behavioral_task: Union[None, ArtificialSubject.Task] = None
        self.task_function_mapping_dict = {
            ArtificialSubject.Task.reading_times: self.estimate_reading_times,
            ArtificialSubject.Task.next_word: self.predict_next_word,
        }

    def identifier(self) -> str:
        return self.model_id

    def start_behavioral_task(self, task: ArtificialSubject.Task):
        self.behavioral_task = task
        self.output_to_behavior = self.task_function_mapping_dict[task]

    def start_neural_recording(
        self,
        recording_target: ArtificialSubject.RecordingTarget,
        recording_type: ArtificialSubject.RecordingType,
    ):
        raise NotImplementedError(
            "Symbolic parsers are probabilistic models that do not support neural tasks."
        )

    def digest_text(self, text: Union[str, List[str]]) -> Dict[str, DataAssembly]:
        """
        :param text: the text to be used for inference e.g. "the quick brown fox"
        :return: assembly of either behavioral output or internal neural representations
        """
        assert self.grammar is not None, "A grammar has not been added to this model"

        self.prefix_probabilities = []

        if isinstance(text, str):
            text = [text]

        output = {"behavior": [], "neural": []}

        # Get the number of tokens in each part
        all_tokens = []
        tokens_per_part = []
        for part in text:
            part_tokens = list(part.split())
            all_tokens += part_tokens
            tokens_per_part.append(len(part_tokens))

        # Tokenize the context. Replace words that don't exist in the grammar with <unk>
        for token_i in range(len(all_tokens)):
            token = all_tokens[token_i]
            if len(self.grammar.productions(rhs=token)) == 0:
                all_tokens[token_i] = "<unk>"

        chart = self.parser.chart_parse(all_tokens)
        dot_position = 0
        for part_number, text_part in enumerate(text):
            context = " ".join(text[: part_number + 1])

            # format output
            stimuli_coords = {
                "stimulus": ("presentation", [text_part]),
                "context": ("presentation", [context]),
                "part_number": ("presentation", [part_number]),
            }

            if self.behavioral_task:
                # format behavioral output into assembly
                behavioral_output = self.output_to_behavior(
                    chart,
                    start=dot_position,
                    end=dot_position + tokens_per_part[part_number],
                )
                behavior = BehavioralAssembly(
                    [behavioral_output], coords=stimuli_coords, dims=["presentation"]
                )
                output["behavior"].append(behavior)

            dot_position += tokens_per_part[part_number]

        # merge over text parts
        self._logger.debug("Merging outputs")
        output["behavior"] = (
            xr.concat(output["behavior"], dim="presentation").sortby("part_number")
            if output["behavior"]
            else None
        )

        return output

    def estimate_reading_times(self, chart: Chart, start: int, end: int) -> float:
        """
        :param chart: a complete chart parse of the input sequence
        :param start: the index of the first token in the current context (inclusive)
        :param end: the index of the last token in the current context (inclusive)
        :return: surprisal (in bits) as a proxy for reading times, following Smith & Levy 2013
            (https://www.sciencedirect.com/science/article/pii/S0010027713000413)
        """
        # Parse context and get parses
        edges = chart.select(end=end)

        # Exclude edges from predict rules
        edges = [p for p in edges if not p.start() == p.end()][1:]

        if not edges:
            # Could not parse the prefix, edge case --> infinite surprisal
            return np.infty

        # Identify the non-terminal parent node, added first by the scanner rule
        nonterminal_parent = edges[0].lhs()
        if len(edges) == 1:
            # Only edge is the scanner edge, so only consider it
            self._add_prefix_probability([edges[0].prob()], [edges[0].span()])
        else:
            # Multiple edges added by completer rule, marginalize over all edges
            probabilities, spans = [], []
            for e in edges:
                if e.dot() > 0 and e.rhs()[e.dot() - 1] == nonterminal_parent:
                    probabilities.append(e.prob())
                    spans.append(e.span())
            self._add_prefix_probability(probabilities, spans)

        p = self.prefix_probabilities[-1]
        return -np.log2(p)

    def predict_next_word(self, chart: Chart, start: int, end: int) -> str:
        """
        :param chart: a complete chart parse of the input sequence
        :param start: the index of the first token in the current context (inclusive)
        :param end: the index of the last token in the current context (inclusive)
        :return: predicted next word
        """
        # Keeps track of non-terminals to avoid infinite recursion
        visited_nonterminals = set()

        def get_next_terminal(grammar: PCFG, lhs: Union[str, Nonterminal]) -> str:
            """
            Given a grammar and a starting left-hand-side (lhs), finds the lexical item
            that is most likely to appear first. Does this by iteratively going through
            the most likely production for the first non-terminal in the rhs. Visited
            non-terminals are pruned to avoid infinite recursion.

            :param grammar: a PCFG grammar containing lhs
            :param lhs: a terminal string or non-terminal node in the grammar
            """

            visited_nonterminals.add(lhs)
            if is_terminal(lhs):
                return lhs

            most_likely_production = max(
                grammar.productions(lhs=lhs),
                key=lambda x: x.prob() if x.rhs()[0] not in visited_nonterminals else 0,
            )
            return get_next_terminal(grammar, most_likely_production.rhs()[0])

        # Parse context and get chart edges
        edges = chart.select(end=end, is_incomplete=True)

        # Exclude edges from predict rules
        edges = [p for p in edges if not p.start() == p.end()]

        if not edges:
            # Context could not be parsed
            next_word = "<unk>"
        elif len(edges) == 1:
            # The only one possible parse under this nonterminal/terminal is the observed word
            next_word = get_next_terminal(self.grammar, edges[0].nextsym())
        else:
            # There are multiple possible parses, so pick the one with the highest probability
            next_lhs = max(edges[:-1], key=lambda x: x.prob()).nextsym()
            next_word = get_next_terminal(self.grammar, next_lhs)

        return next_word

    def create_grammar(
        self,
        treebank_path: str,
        fileids: Union[List[str], str, None] = None,
        unk_low_frequency: bool = True,
        k=2,
    ):
        """
        Creates a PCFG grammar given a path to a treebank corpus (e.g. PTB)

        :param treebank_path: a path to a treebank corpus
        :param grammar_string: one or more file names to be parsed in the grammar. If None, all files will be parsed
        :param unk_low_frequency: if True, replaces all words that appear less than k times by <unk>
        :param k: the <unk> replacement threshold (min number of occurrences for a word to NOT be replaced by <unk>)
        """

        # Load PTB annotations
        treebank = BracketParseCorpusReader(
            treebank_path,
            r".*",
        )

        # First, get all productions and count the occurrences of each lexical in all productions
        productions = []
        lexical_counts = {}
        for tree in treebank.parsed_sents(fileids):
            tree_prods = tree.productions()
            for p in tree_prods:
                for r in p.rhs():
                    if is_terminal(r):
                        lexical_counts[r] = lexical_counts.get(r, 0) + 1
            productions += tree_prods

        # Unkify low-frequency tokens, if flag set to True
        if unk_low_frequency:
            unk_token = "<unk>"
            for p in productions:
                rhs = []
                for r in p.rhs():
                    if is_terminal(r) and lexical_counts[r] < k:
                        rhs.append(unk_token)
                    else:
                        rhs.append(r)
                p._rhs = tuple(rhs)

        # Save grammar
        S = Nonterminal("S")
        self.grammar = nltk.induce_pcfg(S, productions)

        # Save the production probabilities for faster sentence probability estimation
        self.production_probs = {}
        for prod in self.grammar.productions():
            self.production_probs[prod.lhs(), prod.rhs()] = prod.prob()

        # Create parser using the constructed grammar
        self.parser = self.parser_cls(self.grammar, trace=PARSING_TRACE)

    def set_grammar(self, grammar_string: Union[str, None] = None):
        """
        Constructs a PCFG grammar using the provided grammar string and uses it as the instance's grammar.
        If not provided, constructs a new grammar using NLTK's Penn Treebank 02-04.

        :param grammar_string: a grammar string to use as a PCFG, or None to set default grammar
        """

        if grammar_string:
            self.grammar = PCFG.fromstring(grammar_string)
        else:
            # Download NLTK Treebank if doesn't exist
            try:
                nltk.data.find("corpora/treebank")
            except LookupError:
                nltk.download("treebank")
            treebank = nltk.corpus.treebank

            # Use trees to estimate the PCFG
            start = Nonterminal("S")
            productions = []
            for item in treebank.fileids()[2:4]:
                for tree in treebank.parsed_sents(item):
                    tree.collapse_unary()
                    tree.chomsky_normal_form()
                    productions += tree.productions()
            self.grammar = nltk.induce_pcfg(start, productions)

        # Save the production probabilities for faster sentence probability estimation
        self.production_probs = {}
        for prod in self.grammar.productions():
            self.production_probs[prod.lhs(), prod.rhs()] = prod.prob()

        # Create parser using the constructed grammar
        self.parser = self.parser_cls(self.grammar, trace=PARSING_TRACE)

    def _add_prefix_probability(
        self, edge_probabilities: List[float], edge_spans: List[Tuple[int]]
    ):
        if not self.prefix_probabilities:
            # First time adding a partial parse x_0: get P(x_0) by marginalizing over all possible edges
            self.prefix_probabilities.append(sum(edge_probabilities))
            return

        # Adding prefix x_i as the sum of edge probabilities weighted by the its prefix's parse probability
        prefix_probability = 0
        for i in range(len(edge_probabilities)):
            probability = edge_probabilities[i]
            start, end = edge_spans[i]

            prefix_probability += (
                self.prefix_probabilities[start - 1] if start > 0 else 1
            ) * probability

        self.prefix_probabilities.append(prefix_probability)
