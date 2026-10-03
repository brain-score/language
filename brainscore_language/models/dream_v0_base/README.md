# Dream-v0-Base-7B Brain-Score Language plugin

The frozen checkpoint is `Dream-org/Dream-v0-Base-7B` revision
`6572adb5535263e4d1a337b56942ba48b6dee2a9` (Apache-2.0).

Neural: clean passage-local context, independent text-part tokenization,
current-part mean hidden state at layer 21/28, language-system fMRI/ECoG.
Behavior/Engineering: target-hidden prefix-only one-mask surprisal, summed in
bits over region subtokens, fixed 4095 observed-token context and no future
mask canvas. Empty regions contribute zero and are omitted from later context.
The model uses `num_logits_to_keep=1` for behavioral output; a three-length
diagnostic found the same top-1 output and distribution TV below 0.00001
against the full output head. See the frozen research contract for details.

`DREAM_MODEL_PATH` can point to a verified local checkpoint. This package is
not an official Brain-Score result until complete local benchmarks and an
upstream submission are confirmed.
