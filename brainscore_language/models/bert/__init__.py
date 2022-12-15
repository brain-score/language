from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from brainscore_language.model_helpers.huggingface import (
    BidirectionalHuggingfaceSubject,
)

# layer assignment based on choosing the maximally scoring layer on Pereira2018-encoding from
# https://github.com/mschrimpf/neural-nlp/blob/master/precomputed-scores.csv

model_registry["distilgpt2"] = lambda: BidirectionalHuggingfaceSubject(
    model_id="bert-base-uncased",
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: "transformer.h.8"
    },
)
