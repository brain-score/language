# Makefile for `langbrainscore`
# for help, contact asathe@mit.edu

###############################################################################
# default top-level options
###############################################################################
default: help

help: 
	mdless README.md || less README.md

all: build docs test


###############################################################################
# obtain poetry and set up a virtual environment
###############################################################################
poetry:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/1.1.10/get-poetry.py | python3 -

venv: poetry
	poetry shell
###############################################################################


###############################################################################
# building wheels, and installing the project in the current environment
###############################################################################
install: poetry.lock pyproject.toml langbrainscore/**/*.py
	poetry install

build: poetry.lock pyproject.toml langbrainscore/**/*.py
	poetry build
###############################################################################


###############################################################################
# build the docs
###############################################################################
docs: langbrainscore/**/*.py
	pdoc3 --html langbrainscore --force


###############################################################################
# run code coverage, unit tests, and static type checking
###############################################################################
test: coverage typecheck


coverage: langbrainscore/**/*.py tests/*.py
	mkdir -p html test-results

	coverage run -m pytest --junitxml=test-results/tests.xml --html=test-results/tests.html --self-contained-html || true 

	coverage html -d test-results/codecov langbrainscore/**/*.py && rm -f test-results/codecov/.gitignore
	coverage xml -o test-results/codecov.xml langbrainscore/**/*.py 


typecheck: langbrainscore/**/*.py
	mypy -m langbrainscore --config-file pyproject.toml --junit-xml test-results/typing.xml --html-report test-results/typing || true


###############################################################################
# cleanup
###############################################################################
clean:
	/bin/rm -rf html/test-results test-results codecov
	/bin/rm -rf html build dist test-results 