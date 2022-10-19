from brainscore_language import model_registry
from .model import glove

model_registry['glove-840b'] = lambda: glove('glove.840B.300d', dimensions=300)
