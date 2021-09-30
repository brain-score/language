# lang-brainscore
A lang-brainscore repository for initial dirty development; select code will be migrated to a lean and clean more permanent repo in the future.
But for now:


# Installation

## For development

This project uses [`poetry`](https://python-poetry.org/) for dependency management and packaging. 
`poetry` allows running the application in a virtual environment, which can be *either* `conda` *or*
`virtualenv`, (or one of other less commonly used alternatives). 

1. In order to set up your environment, please obtain poetry, a lightweight python package, on your machine.
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

## As an end user

Install this project using PyPI! [`NotImplemented`]
```bash
python3 -m pip install lang-brainscore-fuzzy-potato
```