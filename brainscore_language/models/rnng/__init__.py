from brainscore_language import model_registry, ArtificialSubject
from brainscore_language.model_helpers.container import ContainerSubject


# trained on PTB with 512 hidden units
def make_ptb_512_model(id, rep):
    return ContainerSubject(
        container="benlipkin/rnng:1fd71a2df4a89ddd5f2129d2b7df1088e72ec1cb",
        entrypoint="cd /app && source activate rnng && python -m brainscore",
        identifier=id,
        region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: rep},
    )


# trained on PTB+OANC with 512 hidden units
def make_ptboanc_512_model(id, rep):
    return ContainerSubject(
        container="benlipkin/rnng:70d4c5f95e92c83e5d317073eb92fff458b7aca0",
        entrypoint="cd /app && source activate rnng && python -m brainscore",
        identifier=id,
        region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: rep},
    )


# trained on PTB+OANC with 1024 hidden units
def make_ptboanc_1024_model(id, rep):
    return ContainerSubject(
        container="benlipkin/rnng:2a99249ab6a306e7355852e50a20dcc9cc7757d0",
        entrypoint="cd /app && source activate rnng && python -m brainscore",
        identifier=id,
        region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: rep},
    )


# IMPORTANT NOTE:
# RNNG Models are setup to only support inference over decontextualized sentences.
# e.g., they operate on "text" directly instead of operating on "context".
# For comparison purposes, a comparable RNNLM model is also provided.
# These models are also only setup to support the "neural" task and not any behavioral tasks.
model_registry["rnn-slm-ptb"] = lambda: make_ptb_512_model(
    "rnn-slm-ptb",
    "lstm-mean",
)
model_registry["rnn-tdg-ptb"] = lambda: make_ptb_512_model(
    "rnn-tdg-ptb",
    "rnng-mean",
)
model_registry["rnn-slm-ptboanc"] = lambda: make_ptboanc_512_model(
    "rnn-slm-ptboanc",
    "lstm-mean",
)
model_registry["rnn-tdg-ptboanc"] = lambda: make_ptboanc_512_model(
    "rnn-tdg-ptboanc",
    "rnng-mean",
)
model_registry["rnn-slm-ptboanc-1024"] = lambda: make_ptboanc_1024_model(
    "rnn-slm-ptboanc-1024",
    "lstm-mean",
)
model_registry["rnn-tdg-ptboanc-1024"] = lambda: make_ptboanc_1024_model(
    "rnn-tdg-ptboanc-1024",
    "rnng-mean",
)
