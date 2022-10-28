.. _glossary:

********
Glossary
********

.. glossary::

    Dataset
        A dataset typically consists of either neuroscience data (behavioral or neural recordings)
        or a dataset without any human performance recording (in that case, we term the dataset as an engineering dataset).
        The contents and format of a dataset are not restrictive; in the minimal case, a dataset can consist of just stimuli.

    Metric
        A metric quantifies the similarity between two sets of measurements.
        Typically these are biological and model measurements. Examples of metrics include regression
        (e.g., fitting a regression model from model measurements to predict neural activity),
        representational similarity (comparing representational dissimilarity matrices derived from respectively models and neural representations),
        or correlation of e.g., human and model reading times.
        `Metric specification <https://brain-score-core.readthedocs.io/en/latest/modules/metrics.html>`

    Model
        A model usually refers to a computational model,
        but we generally try to be more explicit in what we refer to with the ArtificialSubject term
        (which can be viewed as an umbrella term for computational models).

    ArtificialSubject
        An ArtificialSubject implements a set of interface functions that benchmarks can interact with.
        This way, we can run experiments on computational models implementing these functions
        in the same way as a human experimental subject. These interface functions include
        behavioral tasks (e.g., next-word prediction or reading times),
        neural recordings (e.g. fMRI or ECoG recordings in the biological brain which neural network models could implement as layer-wise unit activations),
        and a method to digest stimuli.
        :doc:`ArtificialSubject <./artificial_subject>`

    Benchmark
        A benchmark runs an experiment on an ArtificialSubject,
        and compares the resulting measurements (often model predictions) to biological measurements using a particular metric,
        resulting in a similarity score.
        :doc:`Benchmarks <https://brain-score-core.readthedocs.io/en/latest/modules/_autosummary/brainscore_core.benchmarks.html#module-brainscore_core.benchmarks>`

    Plug-in
        Plug-ins are ways of flexibly adding new data, metrics, benchmarks, or models to the Brain-Score platform.

    Xarray
        `Xarray <https://docs.xarray.dev/en/stable/>` DataArrays are multidimensional pandas dataframes.
        DataArrays enable us to retain metadata along several dimensions of the data at once (often needed in neuroscience data, e.g., keeping track of stimuli and neural dimensions).

    Target
        A target is a term for the data that is to be predicted in regression-based metrics;
        the target would be what the regression model is fitted to predict.

    Source
        A source is a term for the data that is used as the predictor in regression-based metrics;
        the source would be what the regression model is fitted on to predict the target.



