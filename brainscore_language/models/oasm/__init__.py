from brainscore_language import model_registry
from .model import OASMSubject

model_registry['oasm'] = lambda: OASMSubject(identifier='oasm')
