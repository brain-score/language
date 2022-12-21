"""
Modified rule definitions for the NLTK abstract chart rules to work with a probabilistic context-free grammar.
Added a probabilstic Earley chart parser by applying incremental chart parsing with the probabilistic rules.
Adapted from: https://www.nltk.org/api/nltk.parse.chart.html
"""

from nltk.grammar import is_nonterminal, is_terminal

from nltk.parse.pchart import (
    ProbabilisticTree,
    AbstractChartRule,
    ProbabilisticTreeEdge,
    ProbabilisticLeafEdge,
    ProbabilisticFundamentalRule,
)

from nltk.parse.chart import Chart
from nltk.parse import IncrementalChartParser
from nltk import PCFG


class ProbabilisticLeafInitRule(AbstractChartRule):
    NUM_EDGES = 0

    def apply(self, chart: Chart, grammar: PCFG):
        # For each token in the sentence
        for index in range(chart.num_leaves()):
            # Create a probabilistic edge at the token
            new_edge = ProbabilisticLeafEdge(chart.leaf(index), index)
            # Insert new edge into the chart if new and yield it
            if chart.insert(new_edge, ()):
                yield new_edge


class ProbabilisticTopDownInitRule(AbstractChartRule):
    r"""
    A rule licensing edges corresponding to the probabilistic grammar
    productions for the grammar's start symbol.
    """

    NUM_EDGES = 0

    def apply(self, chart, grammar):
        # For each production that's S --> ...
        for prod in grammar.productions(lhs=grammar.start()):
            # Create a new edge with index 0
            new_edge = ProbabilisticTreeEdge.from_production(prod, 0, prod.prob())
            # Insert new edge into the chart if new and yield it
            if chart.insert(new_edge, ()):
                yield new_edge


class CompleteProbabilisticFundamentalRule(AbstractChartRule):
    NUM_EDGES = 1

    _fundamental_rule = ProbabilisticFundamentalRule()

    def apply(self, chart, grammar, edge1):
        fr = self._fundamental_rule
        if edge1.is_incomplete():
            # edge1 = left_edge; edge2 = right_edge
            end = edge1.end()
            for edge2 in chart.select(
                start=edge1.end(), end=end, is_complete=True, lhs=edge1.nextsym()
            ):
                yield from fr.apply(chart, grammar, edge1, edge2)
        else:
            # edge2 = left_edge; edge1 = right_edge
            for edge2 in chart.select(
                end=edge1.start(), is_complete=False, nextsym=edge1.lhs()
            ):
                yield from fr.apply(chart, grammar, edge2, edge1)

    def __str__(self):
        return "Fundamental Rule"


class ProbabilisticCompleterRule(CompleteProbabilisticFundamentalRule):
    _fundamental_rule = CompleteProbabilisticFundamentalRule()

    def apply(self, chart, grammar, edge):
        if not isinstance(edge, ProbabilisticLeafEdge):
            yield from self._fundamental_rule.apply(chart, grammar, edge)


class ProbabilisticScannerRule(CompleteProbabilisticFundamentalRule):
    _fundamental_rule = CompleteProbabilisticFundamentalRule()

    def apply(self, chart, grammar, edge):
        if isinstance(edge, ProbabilisticLeafEdge):
            yield from self._fundamental_rule.apply(chart, grammar, edge)


class ProbabilisticTopDownPredictRule(AbstractChartRule):
    r"""
    A cached rule licensing edges corresponding to the grammar
    productions for the nonterminal following an incomplete edge's
    dot.

    After the first time this rule is applied to an edge with a given
    ``end`` and ``next``, it will not generate any more edges for edges
    with that ``end`` and ``next``.

    If ``chart`` or ``grammar`` are changed, then the cache is flushed.
    """

    NUM_EDGES = 1

    def __init__(self):
        AbstractChartRule.__init__(self)
        self._done = {}

    def apply(self, chart, grammar, edge: ProbabilisticTreeEdge):
        if edge.is_complete():
            return
        nextsym, index = edge.nextsym(), edge.end()
        if not is_nonterminal(nextsym):
            return

        # If we've already applied this rule to an edge with the same
        # next & end, and the chart & grammar have not changed, then
        # just return (no new edges to add).
        done = self._done.get((nextsym, index), (None, None))
        if done[0] is chart and done[1] is grammar:
            return

        # Add all the edges indicated by the top down expand rule.
        for prod in grammar.productions(lhs=nextsym):
            # If the left corner in the predicted production is
            # leaf, it must match with the input.
            if prod.rhs():
                first = prod.rhs()[0]
                if is_terminal(first):
                    if index >= chart.num_leaves() or first != chart.leaf(index):
                        continue

            new_edge = ProbabilisticTreeEdge.from_production(prod, index, prod.prob())
            if chart.insert(new_edge, ()):
                yield new_edge

        # Record the fact that we've applied this rule.
        self._done[nextsym, index] = (chart, grammar)


PROBABILISTIC_EARLEY_STRATEGY = [
    ProbabilisticLeafInitRule(),
    ProbabilisticTopDownInitRule(),
    ProbabilisticCompleterRule(),
    ProbabilisticScannerRule(),
    ProbabilisticTopDownPredictRule(),
]


class ProbabilisticEarleyChartParser(IncrementalChartParser):
    def __init__(self, grammar, **parser_args):
        IncrementalChartParser.__init__(
            self, grammar, PROBABILISTIC_EARLEY_STRATEGY, **parser_args
        )

    def parse(self, tokens):
        chart = self.chart_parse(tokens)
        return iter(chart.parses(self._grammar.start(), tree_class=ProbabilisticTree))
