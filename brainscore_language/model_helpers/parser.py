import logging
from typing import Tuple, Union, List, Dict
from pathlib import Path

import numpy as np
from tqdm import tqdm
import xarray as xr
import nltk
from nltk import PCFG
from nltk import Tree, ProbabilisticTree, Nonterminal

from brainio.assemblies import DataAssembly, BehavioralAssembly
from brainscore_language import model_registry
from brainscore_language.artificial_subject import ArtificialSubject
from brainscore_language.utils import fullname

PARSING_TRACE = 0  # how verbose the tracing output should be while parsing a text


def set_tree_probability(tree: ProbabilisticTree, production_probs: Dict[Tuple, float]):
    """
    Calculates the probability of a parsed tree given a set of production probabilities.
    Mutates the Tree instance to contain its production probabilities.
    """
    if tree.prob() is not None:
        return

    # Get the prob of the CFG production.
    lhs = Nonterminal(tree.label())
    rhs = []
    for child in tree:
        if isinstance(child, Tree):
            rhs.append(Nonterminal(child.label()))
        else:
            rhs.append(child)
    prob = production_probs[lhs, tuple(rhs)]

    # Get the probs of children.
    for child in tree:
        if isinstance(child, Tree):
            set_tree_probability(child, production_probs)
            prob *= child.prob()
    tree.set_prob(prob)


class ProbabilisticParserSubject(ArtificialSubject):
    def __init__(
        self,
        model_id: str,
        parser_cls: nltk.ParserI,
        grammar_string: Union[str, None] = None,
    ):
        """
        :param model_id: the model id i.e. name
        """
        self._logger = logging.getLogger(fullname(self))
        self.model_id = model_id
        self.parser_cls = parser_cls

        self.set_grammar(grammar_string)

        self.behavioral_task: Union[None, ArtificialSubject.Task] = None
        self.task_function_mapping_dict = {
            ArtificialSubject.Task.reading_times: self.estimate_reading_times,
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

            # Parse context and get parses
            parses = chart.select(start=dot_position, end=dot_position+tokens_per_part[part_number])
            parses = list(parses)
            dot_position += tokens_per_part[part_number]

            # Assign probabilities to the parsed trees.
            for tree in parses:
                set_tree_probability(tree, self.production_probs)

            # format output
            stimuli_coords = {
                "stimulus": ("presentation", [text_part]),
                "context": ("presentation", [context]),
                "part_number": ("presentation", [part_number]),
            }

            if self.behavioral_task:
                # format behavioral output into assembly
                behavioral_output = self.output_to_behavior(parses)
                behavior = BehavioralAssembly(
                    [behavioral_output], coords=stimuli_coords, dims=["presentation"]
                )
                output["behavior"].append(behavior)

        # merge over text parts
        self._logger.debug("Merging outputs")
        output["behavior"] = (
            xr.concat(output["behavior"], dim="presentation").sortby("part_number")
            if output["behavior"]
            else None
        )

        return output

    def estimate_reading_times(self, parses: List[Tree]):
        """
        :param parses: a list of parsing trees sorted b
        :return: surprisal (in bits) of the most likely parsing of the text
            (https://www.sciencedirect.com/science/article/pii/S0010027713000413)
        """
        if not parses:
            return np.infty
        most_likely_parse = max(parses, key=lambda x: x.prob())
        return -np.log2(most_likely_parse.prob())

    def set_grammar(self, grammar_string=None) -> PCFG:
        """
        Constructs a PCFG grammar using the provided grammar string. If not provided,
        constructs a new grammar using Penn Treebank 02-04.
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
            for item in treebank.fileids():
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
        
