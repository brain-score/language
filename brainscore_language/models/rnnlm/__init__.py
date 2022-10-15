from brainscore_language import model_registry, ArtificialSubject
from brainscore_language.model_helpers.container import ContainerSubject


model_registry["rnn-lm-ptb"] = lambda: ContainerSubject(
    container="benlipkin/rnng:b3c1f7972ec1a278337b951e0e7b07671e7f3e40",
    entrypoint="cd /app && source activate rnng && python -m brainscore",
    identifier="rnn-lm-ptb",
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: "lstm-last"
    },
)
