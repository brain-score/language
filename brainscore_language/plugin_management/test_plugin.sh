#!/bin/bash

PLUGIN_PATH=$1
PLUGIN_NAME=$2
HAS_REQUIREMENTS=$3
PLUGIN_REQUIREMENTS_PATH=$PLUGIN_PATH/requirements.txt
PLUGIN_TEST_PATH=$PLUGIN_PATH/test.py
SINGLE_TEST=$4

echo "${PLUGIN_NAME/_//}"

eval "$(command conda 'shell.bash' 'hook' 2>/dev/null)"
conda create -n $PLUGIN_NAME python=3.8 -y
conda activate $PLUGIN_NAME
if $HAS_REQUIREMENTS; then
  pip install -r $PLUGIN_REQUIREMENTS_PATH
fi

python -m pip install -e ".[test]"

if [ "$SINGLE_TEST" != False ]; then
  echo "Running ${SINGLE_TEST}"
  pytest -m "not requires_gpu and not memory_intense and not slow and not travis_slow" "-vv" $PLUGIN_TEST_PATH "-k" $SINGLE_TEST "--log-cli-level=INFO"
else
  pytest -m "not requires_gpu and not memory_intense and not slow and not travis_slow" $PLUGIN_TEST_PATH
fi

exit $?
