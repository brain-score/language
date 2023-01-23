.. _new_benchmark_tutorial:

**********************
New Benchmark Tutorial
**********************

This example walks through adding a new benchmark and scoring existing models on it.
Everything can be developed locally with full access to publicly available models,
but we strongly encourage you to submit your benchmark to Brain-Score to make it accessible to the community,
and to make it into a goalpost that future models can be measured against.

If you haven't already, check out
`other benchmarks <https://github.com/brain-score/language/tree/main/brainscore_language/benchmarks>`_
and the `docs <https://brain-score-language.readthedocs.io>`_.

A benchmark reproduces the experimental paradigm on a model candidate,
and tests model predictions against the experimentally observed data,
using a similarity metric.

In other words, a benchmark consists of three things (each of which is a plugin):

1. experimental paradigm
2. biological data (neural/behavioral)
3. similarity metric

For the biological data and the similarity metric, benchmarks can use previously submitted data and metrics.
I.e., re-combinations are very much valid.

Brain-Score secondarily also hosts benchmarks that do not pertain to neural or behavioral data,
e.g. engineering (ML) benchmarks and other analyses. These benchmarks do not include biological data,
and the metric might be ground-truth accuracy.


1. Package data (optional)
==========================

You can contribute new data by submitting a data plugin.
If you are building a benchmark using existing data, you can skip this step.

We use the `BrainIO <https://github.com/brain-score/brainio>`_ format to organize data.
Datasets in brainio are called *assemblies* and are based on `xarray <https://xarray.dev>`_,
a multi-dimensional version of pandas, which allows for metadata on numpy arrays of arbitrary dimensionality.

Most assemblies contain a :code:`presentation` dimension for the stimuli that were presented, as well as potentially other
dimensions for e.g. different subjects or different voxels.
The actual measurements (e.g. reading times, or voxel activity) are typically the values of an assembly.

Behavioral data
---------------

The following is an excerpt from the
`Futrell2018 data packaging <https://github.com/brain-score/language/blob/3e6fff2fda528f06cf5ffb3c5474f81acfa91ffe/brainscore_language/data/futrell2018/data_packaging.py>`_.

.. code-block:: python

    from brainio.assemblies import BehavioralAssembly

    reading_times = parse_experiment_data(...)  # load the experimental data, e.g. from .csv files
    # ... obtain as much metadata as we can ...

    assembly = BehavioralAssembly(reading_times, coords={
            'word': ('presentation', voc_word),
            'stimulus_id': ('presentation', stimulus_ID),
            ...
            'subject_id': ('subject', subjects),
            'WorkTimeInSeconds': ('subject', WorkTimeInSeconds_meta),
            ...
            }, dims=('presentation', 'subject'))

Neural data
-----------

The following is an excerpt from the
`Pereira2018 data packaging <https://github.com/brain-score/language/blob/189eed0c2396756fc419115e57633410d0347e59/brainscore_language/data/pereira2018/data_packaging.py>`_.

.. code-block:: python

    from brainio.assemblies import NeuroidAssembly

    neural_recordings = parse_experiment_data(...)  # load the experimental data, e.g. from .mat files
    # ... obtain as much metadata as we can ...

    assembly = NeuroidAssembly(neural_recordings, coords={
           'stimulus': ('presentation', sentences),
           'stimulus_id': ('presentation', stimulus_id),
           ...
           'neuroid_id': ('neuroid', voxel_number),
           'atlas': ('neuroid', atlases),
           ...
           }, dims=['presentation', 'neuroid'])

Register the data plugin
------------------------

So that your data can be accessed via an identifier, you need to define an endpoint in the plugin registry.

For instance, if your data is on S3, the plugin might look as follows:

.. code-block:: python

    from brainscore_language.utils.s3 import load_from_s3

    def load_assembly() -> BehavioralAssembly:
        assembly = load_from_s3(
            identifier="Futrell2018",
            version_id="MpR.gIXN8UrUnqwQyj.kCrh4VWrBvsGf",
            sha1="381ccc8038fbdb31235b5f3e1d350f359b5e287f")
        return assembly

    data_registry['Futrell2018'] = load_assembly

Unit tests
----------

To ensure the data is in the right format, and not corrupted by any future changes, we require all plugins to include
an accompanying :code:`test.py` file with unit tests.

For instance, here is a small unit test example validating the dimensions of a reading times dataset.


.. code-block:: python

    from brainscore_language import load_dataset

    def test_shape(self):
        assembly = load_dataset('Futrell2018')
        assert len(assembly['presentation']) == 10256
        assert len(assembly['subject']) == 180

These unit tests guarantee the continued validity of your plugin, so we encourage rigorous testing methods.


2. Create metric (optional)
===========================

You can contribute a new metric by submitting a metric plugin.
If you are building a benchmark using an existing metric, you can skip this step.

Metrics compute the similarity between two measurements.
These can be model-vs-human, human-vs-human, or model-model.
Measurements could for instance be reading times, or fMRI recordings.

A simple metric could be the pearson correlation of two measurements:

.. code-block:: python

    import numpy as np
    from scipy.stats import pearsonr
    from brainio.assemblies import DataAssembly
    from brainscore_core.metrics import Metric, Score

    class PearsonCorrelation(Metric):
        def __call__(self, assembly1: DataAssembly, assembly2: DataAssembly) -> Score:
            rvalue, pvalue = pearsonr(assembly1, assembly2)
            score = Score(np.abs(rvalue))  # similarity score between 0 and 1 indicating alignment of the two assemblies
            return score

    metric_registry['pearsonr'] = PearsonCorrelation

This is a very simple example and ignores e.g. checks ensuring the ordering is the same, cross-validation,
or keeping track of metadata.

Unit tests
----------

As with all plugins, please provide a :code:`test.py` file to ensure the continued validity of your metric.
For instance, the following is an excerpt from the
`pearson correlation tests <https://github.com/brain-score/language/blob/3e6fff2fda528f06cf5ffb3c5474f81acfa91ffe/brainscore_language/metrics/pearson_correlation/test.py>`_.

.. code-block:: python

    from brainscore_language import load_metric

    def test_weak_correlation():
        a1 = [1, 2, 3, 4, 5]
        a2 = [3, 1, 6, 1, 2]
        metric = load_metric('pearsonr')
        score = metric(a1, a2)
        assert score == approx(.152, abs=.005)


3. Build the benchmark
======================

With data and metric in place, you can put the two together to build a benchmark that scores model similarity to
behavioral or neural measurements.

Structure
---------

A benchmark runs the experiment on a (model) subject candidate in the :code:`__call__` method,
and compares model predictions against experimental data.
All interactions with the model are via methods defined in the :doc:`ArtificialSubject <./artificial_subject>` interface
-- this allows all present and future models to be tested on your benchmark.

For example:

.. code-block:: python

    from brainscore_core.benchmarks import BenchmarkBase
    from brainscore_language import load_dataset, load_metric, ArtificialSubject

    class MyBenchmark(BenchmarkBase):
        def __init__(self):
            self.data = load_dataset('mydata')
            self.metric = load_metric('pearsonr')
            ...

        def __call__(self, candidate: ArtificialSubject) -> Score:
            candidate.start_behavioral_task(ArtificialSubject.Task.reading_times)  # or any other task
            # or e.g. candidate.start_start_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
            #                                            recording_type=ArtificialSubject.RecordingType.fMRI)
            predictions = candidate.digest_text(stimuli)['behavior']
            raw_score = self.metric(predictions, self.data)
            score = ceiling_normalize(raw_score, self.ceiling)
            return score


Behavioral benchmark
--------------------

To test for behavioral alignment, benchmarks compare model outputs to human behavioral measurements.
The model is instructed to perform a certain task (e.g. output reading times), and then prompted to digest text input,
for which it will output behavioral predictions.

For instance, here is a sample excerpt from the
`Futrell2018 benchmark <https://github.com/brain-score/language/blob/85afdae5294d0613fb51c33333aa76c52fc0849e/brainscore_language/benchmarks/futrell2018/__init__.py>`_
comparing reading times:

.. code-block:: python

    class Futrell2018Pearsonr(BenchmarkBase):
        ...

        def __call__(self, candidate: ArtificialSubject) -> Score:
            candidate.start_behavioral_task(ArtificialSubject.Task.reading_times)
            stimuli = self.data['stimulus']
            predictions = candidate.digest_text(stimuli.values)['behavior']
            raw_score = self.metric(predictions, self.data)
            score = ceiling_normalize(raw_score, self.ceiling)
            return score

    benchmark_registry['Futrell2018-pearsonr'] = Futrell2018Pearsonr

Neural benchmark
----------------

To test for neural alignment, benchmarks compare model internals to human internal neural activity,
measured e.g. via fMRI or ECoG.
Running the experiment on the model subject, the benchmark first instructs where and how to perform neural recording,
and then prompts the subject with text input, for which the model will output neural predictions.

For instance, here is a sample excerpt from the
`Pereira2018 linear-predictivity benchmark <https://github.com/brain-score/language/blob/85afdae5294d0613fb51c33333aa76c52fc0849e/brainscore_language/benchmarks/pereira2018/__init__.py#L55>`_
linearly comparing fMRI activity:

.. code-block:: python

    class Pereira2018Linear(BenchmarkBase):
        ...

        def __call__(self, candidate: ArtificialSubject) -> Score:
            candidate.start_neural_recording(recording_target=ArtificialSubject.RecordingTarget.language_system,
                                               recording_type=ArtificialSubject.RecordingType.fMRI)
            stimuli = self.data['stimulus']
            predictions = candidate.digest_text(stimuli.values)['neural']
            raw_score = self.metric(predictions, self.data)
            score = ceiling_normalize(raw_score, self.ceiling)
            return score

    benchmark_registry['Pereira2018-linear'] = Pereira2018Linear

Ceiling
-------

You might have noticed that model alignment scores are always relative to a ceiling.
The ceiling is an estimate of how well the "perfect model" would perform.
Often, this is an estimate of how well an average human is aligned to the specific data.

For instance, the `Pereira2018 ceiling <https://github.com/brain-score/language/blob/85afdae5294d0613fb51c33333aa76c52fc0849e/brainscore_language/benchmarks/pereira2018/ceiling_packaging.py#L52>`_
compares the linear alignment (i.e. using the same metric) of n-1 subjects to a heldout subject.
The `Futrell2018 ceiling <https://github.com/brain-score/language/blob/85afdae5294d0613fb51c33333aa76c52fc0849e/brainscore_language/benchmarks/futrell2018/__init__.py#L59>`_
compares how well one half of subjects is aligned to the other half of subjects,
again using the same metric that is used for model comparisons.

Running models on your benchmark
--------------------------------

You can now locally run models on your benchmark
(see `4. Submit to Brain-Score`_ for running models on the Brain-Score platform).
Run the `score function <https://brain-score-language.readthedocs.io/en/latest/index.html#brainscore_language.score>`_,
passing in the desired model identifier(s) and the identifier for your benchmark.

For instance, you might run:

.. code-block:: python

    from brainscore_language import score

    model_score = score(model_identifier='distilgpt2', benchmark_identifier='benchmarkid-metricid')

Unit tests
----------

As with all plugins, please provide a :code:`test.py` file to ensure the continued validity of your benchmark.
For instance, the following is an excerpt from the
`Futrell2018 tests <https://github.com/brain-score/language/blob/85afdae5294d0613fb51c33333aa76c52fc0849e/brainscore_language/benchmarks/futrell2018/test.py>`_:

.. code-block:: python

    from brainscore_language import ArtificialSubject, load_benchmark

    class DummyModel(ArtificialSubject):
        def __init__(self, reading_times):
            self.reading_times = reading_times

        def digest_text(self, stimuli):
            return {'behavior': BehavioralAssembly(self.reading_times, coords={
                                        'context': ('presentation', stimuli),
                                        'stimulus_id': ('presentation', np.arange(len(stimuli)))},
                                    dims=['presentation'])}

        def start_behavioral_task(self, task: ArtificialSubject.Task):
            if task != ArtificialSubject.Task.reading_times:
                raise NotImplementedError()

    def test_dummy_bad():
        benchmark = load_benchmark('Futrell2018-pearsonr')
        reading_times = RandomState(0).random(10256)
        dummy_model = DummyModel(reading_times=reading_times)
        score = benchmark(dummy_model)
        assert score == approx(0.0098731 / .858, abs=0.001)

    def test_ceiling():
        benchmark = load_benchmark('Futrell2018-pearsonr')
        ceiling = benchmark.ceiling
        assert ceiling == approx(.858, abs=.0005)
        assert ceiling.raw.median('split') == ceiling
        assert ceiling.uncorrected_consistencies.median('split') < ceiling

Benchmark Card Creation
-----------------------

Please include a :code:`README.md` file along with your benchmark to aid users with understanding and implementation.
As part of your README.md file, please include a YAML section using the following format as a guideline:

.. code-block:: yaml
    ---
      benchmark_details:
        developer: <developing individual or organization>
        date: <date of benchmark creation>
        questions: <where to send questions>
        version: <version number>
        type: <behavioral or neural>
        license: <license details>
        citations: <citation information if relevant>

      experiment:
        task: <list of ArtificialSubject task values>
        recording: <ArtificialSubject Recording type>
        experiment_card: <reference existing experiment cards>
        bidirectionality: <unidirectional/bidirectional>

      data:
        accessibility: <public or private>
        modality: <behavioral/neural and modality, e.g. neural; fMRI
        granularity: <neural data granularity>
        method: <how was the data obtainede?>
        references: <abbreviated Bibtex>

      metric:
        mapping: <e.g., RidgeCV, LinReg, RSA>
        metric: <e.g. PearsonR, accuracy>
        crossvalstrat: <cross validation stratification, e.g. passage>
        crossvalsplitcoord: <cross validation split coordinate <e.g. sentence>

      ethical_considerations: <any relevant ethical consideration>

      recommendations: <any relevant caveats and recommendations>

      example_usage: <one example should be in test_integration.py>
    ---

4. Submit to Brain-Score
========================

To share your plugins (data, metrics, and/or benchmarks) with the community
and to make them accessible for continued model evaluation,
please submit them to the platform.

There are two main ways to do that:

1. By uploading a zip file on the website
2. By submitting a github pull request with the proposed changes

Both options result in the same outcome: your plugin will automatically be tested,
and added to the codebase after it passes tests.

Particulars on data
-------------------

To make data assemblies accessible for Brain-Score model evaluations, it needs to be uploaded.
You can self-host your data (e.g. on S3/OSF), or contact us to host your data on S3.
You can also choose to keep your data private such that models can be scored, but the data cannot be accessed.

For uploading data to S3, see the :code:`upload_data_assembly`
in `utils/s3 <https://github.com/brain-score/language/blob/main/brainscore_language/utils/s3.py>`_.
