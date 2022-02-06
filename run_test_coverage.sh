#!/usr/bin/env bash
set -x

rm -r html test-results htmlcov
mkdir -p html htmlcov test-results

coverage run -m pytest --junitxml=test-results/pytest.xml --html=test-results/index.html --self-contained-html || true # junit.xml
coverage html langbrainscore/**/*.py
mv htmlcov html/

mypy -m langbrainscore --config-file pyproject.toml --junit-xml test-results/mypy.xml --html-report test-results/typecheck || true
mv test-results html/

ls -lah html html/test-results