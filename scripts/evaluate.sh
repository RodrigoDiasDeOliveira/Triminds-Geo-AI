#!/bin/bash

echo "========================================"
echo "📊 Starting Evaluation"
echo "========================================"

export PYTHONPATH=$(pwd)

MODEL_PATH="models/best_model.pth"

python src/evaluation/inference.py \
    --model $MODEL_PATH

python src/evaluation/confusion_matrix.py

python src/evaluation/explainability.py

echo "========================================"
echo "✅ Evaluation Finished"
echo "========================================"