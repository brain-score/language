# Language interface migration to UMI

ArtificialSubject and digest_text are the legacy language API. They remain
available for existing plugins and are adapted when loaded through the unified
registry.

For new cross-domain work:

| Legacy language API | Unified Model Interface |
| --- | --- |
| ArtificialSubject or BrainModel | Subject or BrainScoreModel |
| digest_text(text) | process(stimuli) |
| start_behavioral_task(...) | start_task(TaskContext(...)) |
| language-only score command | brainscore.score(model_id, benchmark_id) |

Continue in the distribution's unified/docs/getting_started.md and
unified/docs/umi_api_reference.md. The legacy language docs remain useful for
domain-specific benchmark and submission details.
