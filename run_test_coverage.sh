#!/usr/bin/env bash
set -x
coverage run -m pytest --junitxml=test-results/pytest.xml --html=test-results/pytest.html --self-contained-html || true # junit.xml
coverage report
coverage html
mypy -m langbrainscore --config-file pyproject.toml --junit-xml test-results/mypy.xml --html-report test-results/typecheck || true

mkdir -p html/
mv test-results html/tests
mv htmlcov html/tests/coverage
ls -lah html/tests