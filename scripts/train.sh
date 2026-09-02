#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config/demo.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: config not found: $CONFIG_PATH" >&2
  exit 1
fi

export PYTHONPATH="$(pwd)"

echo "========================================"
echo "Starting Training Pipeline"
echo "Config: $CONFIG_PATH"
echo "========================================"

python src/pipelines/training_pipeline.py --config "$CONFIG_PATH"

echo "========================================"
echo "Training Finished"
echo "========================================"