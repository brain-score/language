# LLaDA-8B-Base Brain-Score Language model plugin

The neural plugin is publicly scored on Brain-Score Language. This update adds
a fixed behavioral operator for Futrell2018 and SyntaxGym.

The frozen neural readout is `GSAI-ML/LLaDA-8B-Base` revision
`0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, using clean inputs, layer 24,
passage-local cumulative context, and the mean hidden state over current text-part
tokens. It supports fMRI and ECoG neural benchmarks. This update leaves the
neural readout unchanged.

For `reading_times`, parts accumulate only within one `digest_text` call.
The available prefix is tokenized independently of the current part so the
target cannot change model-visible BPE tokens. New subtokens are scored
left-to-right using one appended mask (ID 126336), and negative log2
probabilities are summed in bits over the region. Empty SyntaxGym regions
contribute zero and are omitted from later context; spaces before punctuation
are removed as in the official AR adapter. The context budget is 4095 observed
tokens plus one mask. Surprisal is a reading-time proxy, not milliseconds or
an exact diffusion joint likelihood. `next_word` remains unsupported.

The pinned checkpoint's `last_logits_only` path computes only the mask output
head. On three input lengths, it preserves top-1 and has distribution TV below
0.0001 relative to the full head. A 500-word Futrell smoke differed by about
0.000004 in ceiling-normalized score. Future-mask count is an operating-regime
choice: one versus 16 masks changed target surprisal by median 0.63 bits
in absolute value on 13 preselected items. The submitted policy uses one mask.

## Verified scope

Under Brain-Score Language revision `0e0bb4a2d8df5d3c30fe26ec4528f27a188e1cdc`:

- the live LLaDA subject and a fixed GPT-Neo-1.3B control completed all 12
  registered neural identifiers across Pereira2018, Blank2014, Fedorenko2016,
  and Tuckute2024;
- the LLaDA Pereira-384 live raw score (`0.134269`) matched the precomputed
  cache reference (`0.133745`) within the pre-registered `0.001` tolerance;
- this bundle passed Brain-Score plugin discovery via `load_model` and
  two-part live neural digestion at layer 24.

`Tuckute2024-rdm` is not a model-comparable score in this revision because its
official data assembly has one neuroid, making the official row-wise RDM
correlation undefined. It should remain documented rather than averaged into a
neural headline.

Neural and behavioral scores answer different questions. Neither establishes
a diffusion-specific brain mechanism.

## Installation and validation

Place this directory at `brainscore_language/models/llada8b/`. The plugin
registers `llada-8b-base`; Brain-Score's `load_model` assigns that identifier to
the returned object. Set `LLADA_MODEL_PATH` to a verified local checkpoint when
running in an offline or shared-GPU environment; otherwise the plugin resolves
the pinned Hugging Face model ID and revision. The model requires
`trust_remote_code=True` because LLaDA supplies custom model code.

Before external submission, run:

```bash
pytest -q brainscore_language/models/llada8b/test.py
```

and record the exact Brain-Score revision, checkpoint revision, test output,
and benchmark receipts. An input-only audit found 592 cases among 23,445
nonempty Futrell/SyntaxGym parts where tokenizing the completed text first
altered supposedly hidden prefix IDs. Independent prefix tokenization avoids
that leakage.
