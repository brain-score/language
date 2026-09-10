# LLaDA-8B-Base Brain-Score Language model plugin

This is an upstream-ready submission bundle. It has not been externally
submitted to Brain-Score Language.

The frozen neural readout is `GSAI-ML/LLaDA-8B-Base` revision
`0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, using clean inputs, layer 24,
passage-local cumulative context, and the mean hidden state over current text-part
tokens. It supports fMRI and ECoG neural benchmarks only. Behavioral benchmarks
are explicitly deferred pending a registered diffusion conditional-likelihood and
decoding policy.

## Verified scope

Under Brain-Score Language revision `0e0bb4a2d8df5d3c30fe26ec4528f27a188e1cdc`:

- the live LLaDA subject and a fixed GPT-Neo-1.3B control completed all 12
  registered neural identifiers across Pereira2018, Blank2014, Fedorenko2016,
  and Tuckute2024;
- the LLaDA Pereira-384 live raw score (`0.134269`) matched the precomputed
  cache reference (`0.133745`) within the pre-registered `0.001` tolerance;
- this bundle passed Brain-Score plugin discovery via `load_model`, two-part
  live neural digestion at layer 24, and explicit rejection of unsupported
  behavioral tasks.

`Tuckute2024-rdm` is not a model-comparable score in this revision because its
official data assembly has one neuroid, making the official row-wise RDM
correlation undefined. It should remain documented rather than averaged into a
neural headline.

These results calibrate clean representational alignment only. They do not
establish a diffusion-specific mechanism, neural time course, or behavioral
alignment.

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
and any benchmark receipt. Do not add behavioral benchmark claims unless a
separate diffusion behavioral operating regime is implemented and evaluated.
