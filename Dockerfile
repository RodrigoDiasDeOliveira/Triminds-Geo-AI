FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      gdal-bin libgdal-dev \
      libgl1 libglib2.0-0 libgomp1 \
      libexpat1 libspatialindex-dev \
      curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command starts the API. Override with `docker run ... python src/pipelines/training_pipeline.py`
# to run a training job instead.
CMD ["uvicorn", "src.deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]