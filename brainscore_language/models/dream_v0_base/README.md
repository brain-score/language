# Dream-v0-Base-7B Brain-Score Language plugin

The frozen checkpoint is `Dream-org/Dream-v0-Base-7B` revision
`6572adb5535263e4d1a337b56942ba48b6dee2a9` (Apache-2.0).

Neural: clean passage-local context, independent text-part tokenization,
current-part mean hidden state at layer 21/28, language-system fMRI/ECoG.
Behavior/Engineering: target-hidden prefix-only one-mask surprisal, summed in
bits over region subtokens, fixed 4095 observed-token context and no future
mask canvas. Empty regions contribute zero and are omitted from later context.
Dream's native sampler shifts raw logits one position right: the hidden state
at position i predicts token i+1. Behavioral output retains the final two raw
logit positions (`num_logits_to_keep=2`) and reads the preceding observed
position for the appended mask. A mask-only query reads its first raw position,
matching the native shift without introducing a BOS token. The prior
last-position readout was incorrect; earlier behavioral scores require
reevaluation. Clean neural hidden-state extraction is unchanged.

`DREAM_MODEL_PATH` can point to a verified local checkpoint. Local validation
does not establish official Brain-Score scores. Publication requires upstream
merge, official scoring, and confirmation on the public website.
