#!/bin/bash

PLUGIN_PATH=$1
PLUGIN_NAME=$2
HAS_REQUIREMENTS=$3
PLUGIN_REQUIREMENTS_PATH=$PLUGIN_PATH/requirements.txt
PLUGIN_TEST_PATH=$PLUGIN_PATH/test.py

echo $PLUGIN_NAME

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda create -n $PLUGIN_NAME python=3.8 -y
conda activate $PLUGIN_NAME
if $HAS_REQUIREMENTS; then
	pip install -r $PLUGIN_REQUIREMENTS_PATH
else
	echo "Warning: no requirements.txt found. Installing only base dependencies."
fi
pip install poetry
pip install pytest
poetry install
pytest $PLUGIN_TEST_PATH