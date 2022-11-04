from typing import Union, Optional as TOptional, List as TList
from pyparsing import *
import numpy as np

METRICS = {
    'sum': sum,
    'mean': np.mean,
    'median': np.median,
    'range': np.ptp,
    'max': max,
    'min': min
}
# Enable parser packrat (caching)
ParserElement.enablePackrat()

# Relative and absolute tolerance thresholds for surprisal equality
EQUALITY_RTOL = 1e-5
EQUALITY_ATOL = 1e-3


#######
# Define a grammar for prediction formulae.

# References a surprisal region
lpar = Suppress("(")
rpar = Suppress(")")
region = lpar + (Word(nums) | "*") + Suppress(";%") + Word(alphanums + "_-") + Suppress("%") + rpar
literal_float = pyparsing_common.number

class Region:
    def __init__(self, tokens):
        self.region_number = tokens[0]
        self.condition_name = tokens[1]

    def __str__(self):
        return "(%s;%%%s%%)" % (self.region_number, self.condition_name)

    def __repr__(self):
        return f"Region({self.condition_name},{self.region_number})"

    def __call__(self, surprisal_dict):
        if self.region_number == "*":
            return sum(value for (condition, region), value in surprisal_dict.items()
                       if condition == self.condition_name)

        return surprisal_dict[self.condition_name, int(self.region_number)]

class LiteralFloat(object):
    def __init__(self, tokens):
        self.value = float(tokens[0])

    def __str__(self):
        return "%f" % (self.value,)

    def __repr__(self):
        return "LiteralFloat(%f)" % (self.value,)

    def __call__(self, surprisal_dict):
        return self.value

class BinaryOp(object):
    operators: TOptional[TList[str]]

    def __init__(self, tokens):
        self.operator = tokens[0][1]
        if self.operators is not None and self.operator not in self.operators:
            raise ValueError("Invalid %s operator %s" % (self.__class__.__name__,
                                                            self.operator))
        self.operands = [tokens[0][0], tokens[0][2]]

    def __str__(self):
        return "(%s %s %s)" % (self.operands[0], self.operator, self.operands[1])

    def __repr__(self):
        return "%s(%s)(%s)" % (self.__class__.__name__, self.operator, ",".join(map(repr, self.operands)))

    def __call__(self, surprisal_dict):
        op_vals = [op(surprisal_dict) for op in self.operands]
        return self._evaluate(op_vals, surprisal_dict)

    def _evaluate(self, evaluated_operands, surprisal_dict):
        raise NotImplementedError()

class BoolOp(BinaryOp):
    operators = ["&", "|"]
    def _evaluate(self, op_vals, surprisal_dict):
        if self.operator == "&":
            return op_vals[0] and op_vals[1]
        elif self.operator == "|":
            return op_vals[0] or op_vals[1]

class FloatOp(BinaryOp):
    operators = ["-", "+"]
    def _evaluate(self, op_vals, surprisal_dict):
        if self.operator == "-":
            return op_vals[0] - op_vals[1]
        elif self.operator == "+":
            return op_vals[0] + op_vals[1]

class ComparatorOp(BinaryOp):
    operators = ["<", ">", "="]
    def _evaluate(self, op_vals, surprisal_dict):
        if self.operator == "<":
            return op_vals[0] < op_vals[1]
        elif self.operator == ">":
            return op_vals[0] > op_vals[1]
        elif self.operator == "=":
            return np.isclose(op_vals[0], op_vals[1],
                                rtol=EQUALITY_RTOL,
                                atol=EQUALITY_ATOL)

def Chain(op_cls, left_assoc=True):
    def chainer(tokens):
        """
        Create a binary tree of BinaryOps from the given repeated application
        of the op.
        """
        operators = tokens[0][1::2]
        args = tokens[0][0::2]
        if not left_assoc:
            raise NotImplementedError

        arg1 = args.pop(0)
        while len(args) > 0:
            operator = operators.pop(0)
            arg2 = args.pop(0)
            arg1 = op_cls([[arg1, operator, arg2]])

        return arg1

    return chainer

atom = region.setParseAction(Region) | literal_float.setParseAction(LiteralFloat)

prediction_expr = infixNotation(
    atom,
    [
        (oneOf("- +"), 2, opAssoc.LEFT, Chain(FloatOp)),
        (oneOf("< > ="), 2, opAssoc.LEFT, ComparatorOp),
        (oneOf("& |"), 2, opAssoc.LEFT, Chain(BoolOp)),
    ],
    lpar=lpar, rpar=rpar
)


class Prediction(object):
    """
    Predictions state expected relations between language model surprisal
    measures in different regions and conditions of a test suite. For more
    information, see :ref:`architecture`.
    """

    def __init__(self, idx: int, formula: Union[str, BinaryOp], metric: str):
        """
        Args:
            idx: A unique prediction ID. This is only relevant for
                serialization.
            formula: A string representation of the prediction formula, or an
                already parsed formula. For more information, see
                :ref:`architecture`.
            metric: Metric for aggregating surprisals within regions.
        """
        if isinstance(formula, str):
            try:
                formula = prediction_expr.parseString(formula, parseAll=True)[0]
            except ParseException as e:
                raise ValueError("Invalid formula expression %r" % (formula,)) from e

        self.idx = idx
        self.formula = formula

        if metric not in METRICS.keys():
            raise ValueError("Unknown metric %s. Supported metrics: %s" %
                             (metric, " ".join(METRICS.keys())))
        self.metric = metric

    def __call__(self, item):
        """
        Evaluate the prediction on the given item dict representation. For more
        information on item representations, see :ref:`suite_json`.
        """
        # Prepare relevant surprisal dict
        surps = {(c["condition_name"], r["region_number"]): r["metric_value"][self.metric]
                 for c in item["conditions"]
                 for r in c["regions"]}
        return self.apply_prediction_formula(surps)

    def apply_prediction_formula(self, surps):
        return self.formula(surps)

    @classmethod
    def from_dict(cls, pred_dict, idx: int, metric: str):
        """
        Parse from a prediction dictionary representation (see
        :ref:`suite_json`).
        """
        if not pred_dict["type"] == "formula":
            raise ValueError("Unknown prediction type %s" % (pred_dict["type"],))

        return cls(formula=pred_dict["formula"], idx=idx, metric=metric)

    @property
    def referenced_regions(self):
        """
        Get a set of the regions referenced by this formula.
        Each item is a tuple of the form ``(condition_name, region_number)``.
        """
        def traverse(x, acc):
            if isinstance(x, BinaryOp):
                for val in x.operands:
                    traverse(val, acc)
            elif isinstance(x, Region):
                acc.add((x.condition_name, int(x.region_number)))

            return acc

        return traverse(self.formula, set())

    def as_dict(self):
        """
        Serialize as a prediction dictionary representation (see
        :ref:`suite_json`).
        """
        return dict(type="formula", formula=str(self.formula))

    def __str__(self):
        return "Prediction(%s)" % (self.formula,)
    __repr__ = __str__

    def __hash__(self):
        return hash(self.formula)

    def __eq__(self, other):
        return isinstance(other, Prediction) and hash(self) == hash(other)
