"""Dream-v0-Base-7B Brain-Score Language plugin."""

from brainscore_language import model_registry

from .subject import DreamV0BaseSubject


model_registry["dream-v0-base-7b"] = lambda: DreamV0BaseSubject()
