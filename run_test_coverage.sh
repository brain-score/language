#!/usr/bin/env bash
set -x

mkdir -p html/ test-results/

coverage run -m pytest --junitxml=test-results/pytest.xml --html=test-results/pytest.html --self-contained-html || true # junit.xml
coverage html -m langbrainscore/**/*.py
mypy -m langbrainscore --config-file pyproject.toml --junit-xml test-results/mypy.xml --html-report test-results/typecheck || true

mv test-results html/
mv htmlcov html/test-results/

ls -lah html html/test-results