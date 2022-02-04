# langbrainscore`[-fuzzy-potato]`
_the suffix `-fuzzy-potato` (arbitrarily chosen) indicates this project is in the `alpha` stage of software development._

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

### Natural Language Processing (NLP)
researchers may be interested in comparing how similar representations are across various ANN models,
particularly models they develop or study. They may be also interested in creating increasingly more
cognitively plausible models of natural language understanding. Whereas language-brainscore is not a direct
measure of cognitive plausibility of ANN models, it provides a possible direction to optimize towards.

# Documentation
Click on the badge to go to documentation: [![CircleCI](https://circleci.com/gh/language-brainscore/lang-brainscore-fuzzy-potato/tree/main.svg?style=svg)](https://language-brainscore.github.io/lang-brainscore-fuzzy-potato/)


# Usage
(make sure to install the package first. jump to the install section of this README.)

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
## As an end user, or as a library for use in another project
### Installation
Install this project using PyPI (not currently up-to-date)
```bash
python3 -m pip install langbrainscore
```

## Development
### Installation

#### Option A (preferred method)
This project uses [`poetry`](https://python-poetry.org/) for dependency management and packaging
for development purposes (you don't need poetry to install it as a library/package from PyPI). 
Why? `poetry` allows running the application in a virtual environment while abstracting away *which* 
virtual environment you use, e.g.  `conda` or `virtualenv`, (or one of other less common alternatives). 
<!-- In order to use `poetry` within a conda environment, follow step 2 below (and always activate the conda environment prior to using poetry 
within this project). -->

1. In order to set up your environment, obtain [poetry](https://python-poetry.org/docs/master/#installation), a lightweight python package, on your machine.
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
2. [*Optional*] If you want to have a dedicated `conda` environment for this project, create one now (or use an existing one)
    ```bash
    conda create -n lbsfuzpot python=3.8
    conda activate lbsfuzpot
    ```
3. Now use `poetry` to install dependencies
    ```bash
    poetry install
    ```
4. Before running a script using `langbrainscore`, make sure to activate your environment, or type `poetry shell` to create a venv.

#### Option B 

Use a Docker image with all dependencies pre-installed! 
-  `aloxatel/langbrainscore`
-  Click the badge to open the image on Docker hub: [![CircleCI](https://circleci.com/gh/language-brainscore/lang-brainscore-fuzzy-potato/tree/circle-ci.svg?style=svg)](https://hub.docker.com/repository/docker/aloxatel/langbrainscore)


Alternatively, use the `pyproject.toml` file to create your own environment from scratch.
