from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

model_registry['bert-base-uncased'] = lambda: HuggingfaceSubject(model_id='bert-base-uncased', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'bert.encoder.layer.4'}, bidirectional=True)
