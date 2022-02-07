#!/usr/bin/env bash
set -xe

# cleanup to start fresh
rm -rf html/test-results test-results codecov
mkdir -p html test-results

# run pytest using coverage
coverage run -m pytest --junitxml=test-results/tests.xml --html=test-results/tests.html --self-contained-html || true # junit.xml

# generate coverage report
coverage html -d codecov langbrainscore/**/*.py && rm -f codecov/.gitignore
coverage xml -o test-results/codecov.xml langbrainscore/**/*.py 

# run static type checking using mypy
mypy -m langbrainscore --config-file pyproject.toml --junit-xml test-results/typing.xml --html-report test-results/typing || true

# move all the reports and tests into the deployment folder
mv test-results html/
mv codecov html/test-results/

# sanity check in CI
ls -lah html html/test-results