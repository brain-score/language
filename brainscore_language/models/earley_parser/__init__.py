from brainscore_language import model_registry
from brainscore_language.models.earley_parser.parser import EarleyParserSubject

model_registry["earley-parser"] = lambda: EarleyParserSubject(model_id="earley-parser")
