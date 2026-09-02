#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-triminds-geo-ai}"
CONTAINER_NAME="${CONTAINER_NAME:-triminds-geo-ai-api}"
PORT="${PORT:-8000}"

if [[ ! -f "Dockerfile" ]]; then
  echo "Error: Dockerfile not found. Run this script from the repository root." >&2
  exit 1
fi

echo "========================================"
echo "Building Docker image"
echo "Image: $IMAGE_NAME"
echo "========================================"

docker build -t "$IMAGE_NAME" .

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "Starting API container on port $PORT"
docker run -d \
  --name "$CONTAINER_NAME" \
  -p "$PORT:8000" \
  "$IMAGE_NAME"

echo "API container started: $CONTAINER_NAME"
echo "Health endpoint: http://localhost:$PORT/health"