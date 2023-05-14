#!/bin/bash

MODEL_ID=$1
BENCHMARK_ID=$2
ENV_NAME=$3

get_plugin_dir () {
    python brainscore_language/plugin_management/import_plugin print_plugin_dir "$1" "$2"
}

MODEL_DIR=brainscore_language/models/$(get_plugin_dir "models" "$MODEL_ID")
BENCHMARK_DIR=brainscore_language/benchmarks/$(get_plugin_dir "benchmarks" "$BENCHMARK_ID")

MODEL_ENV_YML=$MODEL_DIR/environment.yml
BENCHMARK_ENV_YML=$BENCHMARK_DIR/environment.yml

echo "Setting up conda environment: ${ENV_NAME}"
eval "$(command conda 'shell.bash' 'hook' 2>/dev/null)"
output=$(conda create -n $ENV_NAME python=3.8 -y 2>&1) || echo $output
conda activate $ENV_NAME
if [ -f "$MODEL_ENV_YML" ]; then
  output=$(conda env update --file $MODEL_ENV_YML 2>&1) || echo $output
fi
if [ -f "$BENCHMARK_ENV_YML" ]; then
  output=$(conda env update --file $BENCHMARK_ENV_YML 2>&1) || echo $output
fi
output=$(python -m pip install "." 2>&1) || echo $output

echo "Scoring ${MODEL_ID} on ${BENCHMARK_ID}"
python brainscore_language score --model_identifier=$MODEL_ID --benchmark_identifier=$BENCHMARK_ID

exit $?
