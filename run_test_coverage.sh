#!/usr/bin/env bash
set -x

# cleanup to start fresh
rm -rf html test-results coverage
mkdir -p html test-results

# run pytest using coverage
coverage run -m pytest --junitxml=test-results/tests.xml --html=test-results/tests.html --self-contained-html || true # junit.xml

# generate coverage report
coverage html -d coverage langbrainscore/**/*.py 
coverage xml -o test-results/coverage.xml langbrainscore/**/*.py 

# run static type checking using mypy
mypy -m langbrainscore --config-file pyproject.toml --junit-xml test-results/typing.xml --html-report test-results/typing || true

# move all the reports and tests into the deployment folder
mv test-results html/
mv coverage html/test-results/

# sanity check in CI
ls -lah html html/test-results