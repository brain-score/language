"""Brain-Score Language plugin for the frozen clean LLaDA-8B-Base neural readout."""

from brainscore_language import model_registry

from .subject import LLaDA8BSubject


model_registry['llada-8b-base'] = lambda: LLaDA8BSubject()
