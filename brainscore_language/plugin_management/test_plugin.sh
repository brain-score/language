#!/bin/bash

PLUGIN_PATH=$1
PLUGIN_NAME=$2
HAS_REQUIREMENTS=$3
PLUGIN_REQUIREMENTS_PATH=$PLUGIN_PATH/requirements.txt
PLUGIN_TEST_PATH=$PLUGIN_PATH/test.py
SINGLE_TEST=$4
ENV_PATH=$PLUGIN_NAME

echo "${PLUGIN_NAME/_//}"

if $HAS_REQUIREMENTS; then
	eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
	output=`conda create -n $PLUGIN_NAME python=3.8 -y 2>&1` || echo $output
	conda activate $PLUGIN_NAME
	output=`pip install -r $PLUGIN_REQUIREMENTS_PATH 2>&1` || echo $output
else
	echo "Warning: no requirements.txt found. Running in base environment."
fi

output=`python -m pip install -e ".[test]" 2>&1` || echo $output

if [ "$SINGLE_TEST" != False ]; then
	echo "Running ${SINGLE_TEST}" 
	pytest -m "not requires_gpu and not memory_intense and not slow and not travis_slow" "-vv" $PLUGIN_TEST_PATH "-k" $SINGLE_TEST "--log-cli-level=INFO"
else
	pytest -m "not requires_gpu and not memory_intense and not slow and not travis_slow" $PLUGIN_TEST_PATH
fi

exit $?
