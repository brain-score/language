from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

# layer assignment based on choosing the maximally scoring layer on Pereira2018-encoding from
# https://github.com/mschrimpf/neural-nlp/blob/master/precomputed-scores.csv

model_registry['gpt-neo-2.7B'] = lambda: HuggingfaceSubject(model_id='EleutherAI/gpt-neo-2.7B', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.31'})

model_registry['gpt-neo-1.3B'] = lambda: HuggingfaceSubject(model_id='EleutherAI/gpt-neo-1.3B', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.18'})
