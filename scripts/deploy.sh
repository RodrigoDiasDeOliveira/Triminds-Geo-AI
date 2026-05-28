#!/bin/bash

echo "========================================"
echo "☁️ Starting Deployment"
echo "========================================"

IMAGE_NAME="satellite-land-classification"

docker build -t $IMAGE_NAME .

docker run -d \
    -p 8000:8000 \
    --name satellite-api \
    $IMAGE_NAME

echo "========================================"
echo "✅ API running on port 8000"
echo "========================================"