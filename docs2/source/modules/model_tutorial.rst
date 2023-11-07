.. _new_model_tutorial:

******************
New Model Tutorial
******************

This example walks through adding a new model and scoring it on existing benchmarks.
Everything can be developed locally with full access to publicly available benchmarks,
but we strongly encourage you to submit your model to Brain-Score to make it accessible to the community,
and to make it testable on future benchmarks.

If you haven't already, check out
`other models <https://github.com/brain-score/language/tree/main/brainscore_language/models>`_
and the `docs <https://brain-score-language.readthedocs.io>`_.


Adding the model plugin
=======================

We require models to implement the :doc:`ArtificialSubject API <./artificial_subject>`.
This interface is the central communication point between models and benchmarks
and guarantees that your model can be evaluated on all available benchmarks.
It includes three central methods that set the model up for performing a behavioral task,
for performing neural recordings, and for digesting text with behavioral and/or neural outputs.
A model does not have to implement all three methods, it can for instance only engage on behavior, or only on neurons.

HuggingFace models
------------------

For models on HuggingFace, we provide a simple :code:`HuggingfaceSubject` wrapper that lets you instantiate models in
very few lines of code since the wrapper takes care of implementing
the :doc:`ArtificialSubject <./artificial_subject>` interface.
The main choice you will have to make is which layer corresponds to which brain region.
For instance, the following is an excerpt from adding
`gpt models <https://github.com/brain-score/language/blob/5e948f0be90327aefe5e2938b2b3a193d0109af2/brainscore_language/models/gpt/__init__.py>`_:

.. code-block:: python

    from brainscore_language.model_helpers.huggingface import HuggingfaceSubject

    model_registry['distilgpt2'] = lambda: HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.5.mlp.dropout'})

Unit tests
----------

As with all plugins, please provide a :code:`test.py` file to ensure the continued validity of your model.
For instance, the following is an excerpt from the
`tests for gpt2-xl <https://github.com/brain-score/language/blob/5e948f0be90327aefe5e2938b2b3a193d0109af2/brainscore_language/models/gpt/test.py>`_:

.. code-block:: python

    from brainscore_language import load_model

    def test_next_word(model_identifier, ):
        model = load_model('gpt2-xl')
        text = ['the quick brown fox', 'jumps over', 'the lazy']
        expected_next_words = ['jumps', 'the', 'dog']
        model.perform_behavioral_task(task=ArtificialSubject.Task.next_word)
        next_word_predictions = model.digest_text(text)['behavior']
        np.testing.assert_array_equal(next_word_predictions, expected_next_words)


Running your model on benchmarks
================================

You can now locally run models on your benchmark
(see `Submit to Brain-Score`_ for running models on the Brain-Score platform).
Run the `score function <https://brain-score-language.readthedocs.io/en/latest/index.html#brainscore_language.score>`_,
passing in the desired benchmark identifier(s) and the identifier for your model.

For instance, you might run:

.. code-block:: python

    from brainscore_language import score

    model_score = score(model_identifier='distilgpt2', benchmark_identifier='Futrell2018-pearsonr')


Submit to Brain-Score
=====================

To share your model plugin with the community and to make it accessible for continued benchmark evaluation,
please submit it to the platform.

There are two main ways to do that:

1. By uploading a zip file on the website
2. By submitting a github pull request with the proposed changes

Both options result in the same outcome: your plugin will automatically be tested,
and added to the codebase after it passes tests.
