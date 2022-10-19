from brainscore_language import benchmark_registry
from .benchmark import WikitextAccuracy

benchmark_registry['Wikitext-accuracy'] = WikitextAccuracy
