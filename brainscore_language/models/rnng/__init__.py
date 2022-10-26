from brainscore_language import model_registry, ArtificialSubject
from brainscore_language.model_helpers.container import ContainerSubject


def make_model(id, rep):
    return ContainerSubject(
        container="benlipkin/rnng:9c49531c104b65753cc82187f019534b3f110def",
        entrypoint="cd /app && source activate rnng && python -m brainscore",
        identifier=id,
        region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: rep},
    )


# IMPORTANT WARNING:
# RNNG Models are setup to only support inference over decontextualized sentences.
# e.g., they operate on "text" directly instead of operating on "context"
# and extracting the "text" indices, which is typical of other models in BrainScore.
# For comparison purposes, a comparable RNNLM model is also provided.
# As such, the models below are not directly comparable to other models in BrainScore.
# They are also only setup to support the "neural" task and not any behavioral tasks.
# This is a result of an incompatibility with the current definition of the behavioral tasks,
# which operate over single words of text conditioned on very long contexts.
model_registry["rnn-slm-ptb"] = lambda: make_model("rnn-slm-ptb", "lstm-mean")
model_registry["rnn-tdg-ptb"] = lambda: make_model("rnn-tdg-ptb", "rnng-mean")
model_registry["rnn-lcg-ptb"] = lambda: make_model("rnn-lcg-ptb", "rnng-mean")
