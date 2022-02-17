# langbrainscore
_release stage: "fuzzy-potato" (alpha)_


### [**Documentation** ![CircleCI](https://circleci.com/gh/language-brainscore/lang-brainscore-fuzzy-potato/tree/main.svg?style=svg)](https://language-brainscore.github.io/lang-brainscore-fuzzy-potato/)
### development status indicators
- [Unit Tests](https://language-brainscore.github.io/lang-brainscore-fuzzy-potato/test-results/tests.html): this report lists tests used for the module and their outcomes: success/failure (using [pytest](https://docs.pytest.org/en/7.0.x/))
- [Code Coverage](https://language-brainscore.github.io/lang-brainscore-fuzzy-potato/test-results/codecov/): describes what parts of the code are tested down to individual lines (using [Coverage.py](https://coverage.readthedocs.io/en/6.3.1/))
- [Static Type Checks](https://language-brainscore.github.io/lang-brainscore-fuzzy-potato/test-results/typing/): results of static type-checking of the code where [type annotations](https://www.python.org/dev/peps/pep-0484/) are available (using [Mypy](https://mypy.readthedocs.io/en/stable/))


## what is it?
Provides a library for systematic comparison of encoder representations in the most general sense.
An `encoder` is an entity that `encode`s linguistic input (e.g., text), and returns a representation of it
(typically in high-dimensional space).
We envision encoders to be either human brains or artificial neural networks (ANNs).
Humans see textual stimuli on a screen which leads to certain computations in the brain,
which can be measured using several proxies, such as fMRI, EEG, ECOG. Similarly, ANNs process textual input
in the form of vectors, and output either some sort of embeddings or latent vectors, all
meant to be useful representations of input for downstream tasks.


## why should I care?
In this project, and in this general family of research projects, we want to evaluate the similarity between
various ways of generating representations of input stimuli. We are also interested in eventually understanding
what kind of representations the brain employs, and how we can go closer to these, and building models helps
us travel in that direction.

### Neuroscientists/Cognitive Scientisits
may be interested in developing better models of brain activation to understand what kind of stimuli drive 
response in certain parts of the brain. While similar efforts exist in the vision domain, in this project,
we target language processing in the brain. We provide ways to use several existing fMRI datasets as benchmarks
for computing a language-brainscore. We also provide ways to work with your own data and test ANN models against
this data.

### Natural Language Processing (NLP) and Deep Learning Researchers
researchers may be interested in comparing how similar representations are across various ANN models,
particularly models they develop or study. They may be also interested in creating increasingly more
cognitively plausible models of natural language understanding. Whereas language-brainscore is not a direct
measure of cognitive plausibility of ANN models, it provides a possible direction to optimize towards.


# Usage
(make sure to install the package first: jump to the [install section](https://github.com/language-brainscore/lang-brainscore-fuzzy-potato/edit/main/README.md#option-a-preferred-method) of this README)

This project has examples hosted on binder. Simply click on the binder launch button to view a Jupyter notebook
with example usage.
Alternatively, take a peek at the `examples/` directory for scripts as well as notebooks.

Following is a schematic of the library usage. Note that it is not a minimal working example (MWE). You will
find MWEs in `examples/`.
```python
import langbrainscore as lbs

gpt2 = lbs.encoder.HuggingFaceEncoder(pretrained_model_name_or_path='gpt2')

pereira18_data = ...
brain = lbs.encoder.BrainEncoder(pereira18_data)

for encoder in [brain, gpt2]:
    print(encoder, encoder.encode(pereira18_data).shape)

```


# How to Obtain this Project?
## Stable: As an end user, or as a library for use in another project
### Installation
Install this project using PyPI (not up-to-date; not recommended as of now)
```bash
python3 -m pip install langbrainscore
```

## Development: Bleeding edge version from GitHub
### Installation

#### Option A (preferred method)
This project uses [`poetry`](https://python-poetry.org/) for dependency management and packaging
for development purposes (you don't need poetry to install it as a library/package from PyPI). 
Why? `poetry` allows running the application in a virtual environment while abstracting away *which* 
virtual environment you use, e.g.  `conda` or `virtualenv`, (or one of other less common alternatives). 
<!-- In order to use `poetry` within a conda environment, follow step 2 below (and always activate the conda environment prior to using poetry 
within this project). -->

1. In order to set up your environment, obtain [poetry](https://python-poetry.org/docs/master/#installation), a lightweight python package, on your machine.
    <!-- curl -sSL https://install.python-poetry.org | python3 - -->
    ```bash
    $ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/1.1.10/get-poetry.py | python3 -
        <OR>
    $ make poetry
    ```
2. If you want to have a dedicated `conda` environment for this project, create one now (or use an existing one). Else, let `poetry` create a `venv`.
    ```bash
    (base) $ conda create -n langbrainscore-env python=3.8
    (base) $ conda activate langbrainscore-env
    (langbrainscore-env) $

        <OR>

    $ poetry shell
    (.venv) $

        <OR>

    $ make venv
    (.venv) $
    ```
3. Now use `poetry` to install the package and dependencies by navigating inside the repository 
    ```bash
    (langbrainscore-env) $ poetry install
        <OR>
    (.venv) $ make install
    ```
4. Before running a script using `langbrainscore`, make sure to activate your environment, or type `poetry shell` to create a venv.

#### Option B 

Use a Docker image with the package and all dependencies pre-installed! 
-  `aloxatel/langbrainscore` (Debian-Ubuntu 20.04 derivative)
-  Click the badge to open the image on Docker hub: [![CircleCI](https://circleci.com/gh/language-brainscore/lang-brainscore-fuzzy-potato/tree/circle-ci.svg?style=svg)](https://hub.docker.com/r/aloxatel/langbrainscore)


Alternatively, use the `pyproject.toml` file to create your own environment from scratch.


<br>

## How to interpret `langbrainscore` versions?

We follow the **Semantic Versioning** spec
([`semver.org v2.0.0`](https://semver.org/spec/v2.0.0.html)):
> *Given a version number `MAJOR.MINOR.PATCH`, increment the:*
> - *`MAJOR` version when you make incompatible API changes,*
> - *`MINOR` version when you add functionality in a backwards compatible manner, and*
> - *`PATCH` version when you make backwards compatible bug fixes.*
> *Additional labels for pre-release and build metadata are available as extensions to the `MAJOR.MINOR.PATCH` format.*

Additionally:
> *Major version zero `(0.y.z)` is for initial development. Anything MAY change at any time. The public API SHOULD NOT be considered stable. [[ref]](https://semver.org/spec/v2.0.0.html#spec-item-4).*
