from brainscore_language import model_registry
from brainscore_language.models.earley_parser.parser import EarleyParserSubject

model_registry["earley-parser"] = EarleyParserSubject
