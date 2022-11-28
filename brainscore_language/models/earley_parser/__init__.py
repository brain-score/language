from brainscore_language import model_registry
from brainscore_language.model_helpers.parser import ProbabilisticParserSubject
from brainscore_language.models.earley_parser.utils import EarleyChartParser

model_registry["earley-parser"] = lambda: ProbabilisticParserSubject(
    model_id="earley-parser",
    parser_cls=EarleyChartParser,
)