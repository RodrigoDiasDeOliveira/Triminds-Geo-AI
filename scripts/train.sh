#!/bin/bash

echo "========================================"
echo "🚀 Starting Training Pipeline"
echo "========================================"

export PYTHONPATH=$(pwd)

CONFIG_PATH="config/training.yaml"

python src/pipelines/training_pipeline.py \
    --config $CONFIG_PATH

echo "========================================"
echo "✅ Training Finished"
echo "========================================"