Brain-Score Language
====================

Brain-Score is a collection of benchmarks and models:
benchmarks combine neural/behavioral data with a metric to score models on their alignment to humans,
and models are evaluated as computational hypotheses of human brain processing.

The Brain-Score Language library contains
benchmarks that can easily be used to test language models on their alignment to human behavioral and internal brain
processing,
as well as language models that can easily be tested on new behavioral or neural data.
This makes experimental data accessible to modelers, and computational models accessible to experimenters,
accelerating progress in discovering ever-more-accurate models of the human brain and mind.

The `score` function is the primary entry point to score a model on a benchmark.

.. autofunction:: brainscore_language.score

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules/artificial_subject
   brainscore_language/plugins
   modules/model_tutorial
   modules/benchmark_tutorial
   modules/api_reference
   modules/plugins
