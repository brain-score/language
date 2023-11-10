from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# layer assignment was determined by scoring each transformer layer against three neural
# benchmarks: Pereira2018.243sentences-linear, Pereira2018.384sentences-linear, and
# Blank2014-linear, and choosing the layer with the highest average score.

# BERT
model_registry['bert-base-uncased'] = lambda: HuggingfaceSubject(model_id='bert-base-uncased', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'bert.encoder.layer.4'}, bidirectional=True)
