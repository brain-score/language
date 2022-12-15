from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject,HuggingfaceGroup
from .modeling import get_layer_names

# layer assignment based on choosing the maximally scoring layer on Pereira2018-encoding from
# https://github.com/mschrimpf/neural-nlp/blob/master/precomputed-scores.csv

model_registry['distilgpt2'] = lambda: HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.5.mlp.dropout'})

model_registry['gpt2-xl'] = lambda: HuggingfaceSubject(model_id='gpt2-xl', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.43.mlp.dropout'})

model_registry['distilgpt2-layerwise'] = lambda: HuggingfaceGroup(model_id='distilgpt2', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: get_layer_names('distilgpt2')})
