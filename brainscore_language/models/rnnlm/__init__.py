from brainscore_language import model_registry, ArtificialSubject
from brainscore_language.model_helpers.container import ContainerSubject


model_registry["rnn-lm-ptb"] = lambda: ContainerSubject(
    container="benlipkin/rnng:6f6825d1c4a8c58c844b4b82123b967bb0bab6ce",
    entrypoint="cd /app && source activate rnng && python -m brainscore",
    identifier="rnn-lm-ptb",
    region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: "lstm-mean"
    },
)
