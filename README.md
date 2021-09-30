# lang-brainscore
A lang-brainscore repository for initial dirty development; select code will be migrated to a lean and clean more permanent repo in the future.
But for now:


# How to Obtain this Project?

## As an end user, or in another project
### Installation
Install this project using PyPI! [`NotImplemented`]
```bash
python3 -m pip install lang-brainscore-fuzzy-potato
```

## Development
### Installation

This project uses [`poetry`](https://python-poetry.org/) for dependency management and packaging
for development purposes (you don't need poetry to install it as a library/package from PyPI). 
Why? `poetry` allows running the application in a virtual environment while abstracting away *which* 
virtual environment you use, e.g.  `conda` or `virtualenv`, (or one of many other less commonly used 
alternatives). 
<!-- In order to use `poetry` within a conda environment, follow step 2 below (and always activate the conda environment prior to using poetry 
within this project). -->

1. In order to set up your environment, obtain poetry, a lightweight python package, on your machine.
    ```bash
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
    ```
1. [*Optional*] If you want to have a dedicated `conda` environment for this project, create one now (or use an existing one)
    ```bash
    conda env create -n lbsfuzpot
    conda activate lbsfuzpot
    ```
1. Now use `poetry` to install dependencies
    ```bash
    poetry install
    ```
### Running Scripts & Notebooks

There are multiple ways you can do this:
1. Run the project inside a `poetry shell`!
    - This activates a virtual environment if one isn't already active. If you would like to use a conda environment, activate it: `conda activate lbsfuzpot`
    ```bash
    poetry shell
    ```
1. Run the project using `poetry run`: if executing a single command, `poetry run` will 
spawn the environment, run it, and exit, returning you to your original environment.
    ```bash
    poetry run python -m langbrainscore -h
    ```


# Usage