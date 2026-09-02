# Triminds Geo AI

A modular geospatial AI platform for satellite and Earth Observation workflows, built around PyTorch, geospatial processing, FastAPI, MLOps and Google Cloud integration.

> **Current validation scope:** the repository contains a fully local RGB demo path from synthetic dataset generation through model training, checkpoint creation and FastAPI inference. Google Satellite Embedding, Earth Engine and cloud deployment capabilities are represented as configurable platform components and require their respective data, credentials and infrastructure to be enabled.

## Local Demo — Quick Start

The fastest way to validate the repository is the local demo.

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 2. Train the demo model

```bash
bash scripts/train.sh
```

The script automatically creates a small deterministic **synthetic RGB dataset** when `data/demo/train` is missing. The dataset exists only to validate the software lifecycle; it is not a benchmark and does not represent real satellite imagery.

Training produces a checkpoint under `artifacts/`.

To use another configuration:

```bash
bash scripts/train.sh config/your-config.yaml
```

### 3. Start the API

```bash
uvicorn src.deployment.api.main:app --host 0.0.0.0 --port 8000
```

Then verify:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/classes
```

For inference, upload an image to `POST /predict`.

### 4. Docker

Build and run the API container after a checkpoint has been created:

```bash
bash scripts/deploy.sh
```

The container exposes port `8000` and includes a health check at `/health`.

### 5. Tests and quality checks

```bash
ruff check .
ruff format --check .
pytest
pre-commit run --all-files
```

## Architecture

Triminds is designed as a provider-based geospatial AI platform:

```text
Earth Observation Sources
        │
        ▼
   Provider Layer
        │
        ▼
   Ingestion / Validation
        │
        ▼
 Geo Feature / Representation Layer
        │
        ├───────────────┐
        ▼               ▼
 Deep Learning      Vector / Search
        │
        ▼
     FastAPI
        │
        ▼
 Cloud Deployment
```

The long-term architecture supports Google Satellite Embeddings, Google Earth Engine, Sentinel-2, Landsat and other providers. Optional cloud components include Cloud Storage, Vertex AI, Dataproc, BigQuery and Terraform-managed infrastructure.

## Model Layer

The model factory supports multiple architectures, including ResNet, EfficientNet, Vision Transformers and hybrid configurations. An embedding adapter is available for workflows using 64-channel foundation-model representations.

The **local demo deliberately uses RGB (3-channel) ResNet50**. It is separate from the Google Satellite Embedding configuration and should not be interpreted as validation of the 64-channel embedding pipeline.

## MLOps

MLflow and the local model registry are available as optional components. The local demo can run without MLflow, while cloud-oriented configurations can enable the relevant tracking and artifact services.

## Cloud / GCP

The project includes configuration and infrastructure components for Google Cloud workflows. These are environment-dependent and require project configuration, authentication, permissions, datasets and cloud resources before they can be considered operational.

## Project Status

**Active development / production-readiness audit.**

The repository is intentionally separating:

- **Validated:** local demo lifecycle and API inference path.
- **Implemented components:** model, data, geospatial, MLOps and deployment modules.
- **Cloud-dependent:** Earth Engine, Google Satellite Embeddings and GCP deployment workflows.
- **Planned:** additional providers, vector search expansion and Kubernetes/event-driven capabilities.

## License

MIT License
