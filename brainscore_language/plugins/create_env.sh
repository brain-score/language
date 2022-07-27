#!/bin/bash

PLUGIN_PATH=$1
PLUGIN_NAME=$2
PLUGIN_TEST_PATH=$PLUGIN_PATH/test.py
PLUGIN_REQUIREMENTS_PATH=$PLUGIN_PATH/requirements.txt

echo $PLUGIN_NAME

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda create -n $PLUGIN_NAME python=3.8 -y
conda activate $PLUGIN_NAME
pip install -r $PLUGIN_REQUIREMENTS_PATH
pip install poetry
pip install pytest
poetry install
pytest $PLUGIN_TEST_PATH