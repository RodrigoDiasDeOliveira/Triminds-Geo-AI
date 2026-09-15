# Triminds Geo AI

## Geospatial AI Platform for Earth Observation

**Triminds Geo AI** is a modular geospatial AI platform for satellite and Earth Observation workflows, combining machine learning, geospatial processing, FastAPI and cloud-native deployment.

The platform is designed around a provider-oriented architecture so that geospatial data sources, representation models, inference components and deployment environments can evolve independently.

---

## Operational Deployment

The current platform is deployed and online on **Google Cloud Run**.

| Property | Current state |
|---|---|
| Platform | Google Cloud Run |
| Region | `europe-west1` |
| Deployment | Cloud Run |
| Version | `v4` |
| Status | Online / operational |

**Live service:**

[Triminds Geo AI — Cloud Run v4](https://triminds-geo-ai-v4-1091629879450.europe-west1.run.app/)

The Cloud Run deployment represents the current operational instance of the platform.

---

## Architecture

The platform follows a provider-based geospatial AI architecture:

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
Geospatial Representation
        │
        ├───────────────┐
        ▼               ▼
 Deep Learning      Vector / Search
        │               │
        └───────┬───────┘
                ▼
             FastAPI
                │
                ▼
          Cloud Run
                │
                ▼
       Operational Service
```

The architecture is intended to support multiple Earth Observation providers and representation strategies without coupling the application to a single data source.

---

## Current Deployment vs. Local Validation

The repository contains two complementary execution paths.

### Operational Cloud Deployment

The current production-oriented deployment runs on Google Cloud Run as version **v4**.

This is the environment used to expose the platform as an online service.

### Local Demo

The repository also maintains a deterministic local demonstration path for development and reproducibility.

The local path uses:

- synthetic RGB data
- PyTorch
- ResNet50
- local model training
- checkpoint generation
- FastAPI inference

The synthetic dataset is intentionally a software-validation mechanism. It should not be interpreted as a benchmark or as real satellite imagery.

---

## Local Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 2. Train the demo model

```bash
bash scripts/train.sh
```

The training script creates a small deterministic synthetic RGB dataset when the demo data is unavailable.

Training produces a checkpoint under:

```text
artifacts/
```

### 3. Start the local API

```bash
CONFIG_PATH=config/demo.yaml uvicorn src.deployment.api.main:app --host 0.0.0.0 --port 8000
```

Windows PowerShell:

```powershell
$env:CONFIG_PATH="config/demo.yaml"
uvicorn src.deployment.api.main:app --host 0.0.0.0 --port 8000
```

Verify:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/classes
```

Inference:

```text
POST /predict
```

---

## Containerization

The API can also be executed as a container.

```bash
bash scripts/deploy.sh
```

The container exposes port `8000` and provides a health-check endpoint.

---

## Model Layer

The model factory supports multiple architectures, including:

- ResNet
- EfficientNet
- Vision Transformers
- hybrid configurations

An embedding adapter is available for workflows based on higher-dimensional foundation-model representations.

The local demonstration deliberately uses:

```text
RGB → ResNet50
```

This local path is independent from specialized satellite-embedding configurations.

---

## Earth Observation Direction

The architecture is designed to accommodate Earth Observation sources and representation systems such as:

- Google Satellite Embeddings
- Google Earth Engine
- Sentinel-2
- Landsat
- additional providers

The objective is to maintain a modular boundary between data acquisition, representation, machine learning and inference.

---

## MLOps

The repository includes MLOps-oriented components such as:

- MLflow integration
- local model registry
- model artifacts
- configurable training pipelines

These components can be enabled according to the deployment environment.

---

## Google Cloud

Google Cloud is now part of the **operational deployment path**, not only a future integration target.

The current online deployment runs on:

```text
Google Cloud
      │
      ▼
Cloud Run
      │
      ▼
Triminds Geo AI v4
      │
      ▼
Online Service
```

Additional Google Cloud services can be introduced according to the requirements of specific geospatial workloads.

---

## Engineering and Operational Status

The project should be understood through separate validation levels:

| Area | Status |
|---|---|
| Local deterministic demo | Validated |
| FastAPI inference path | Validated |
| Container execution | Validated |
| Google Cloud integration | Implemented |
| Cloud Run deployment | **Operational — v4** |
| Online service | **Available** |
| Additional Earth Observation providers | Evolutionary |
| Advanced geospatial workloads | Under continuous development |

The important distinction is that **Cloud Run v4 is an actual operational deployment**, while the platform itself continues to evolve toward broader operational maturity.

---

## Production Readiness

Triminds Geo AI follows the Triminds engineering principle of separating implementation from operational evidence.

A component may exist in the codebase without being considered operational.

Conversely, the Cloud Run v4 deployment provides concrete evidence that the platform can be deployed and exposed as an online service.

The maturity model is therefore:

```text
Implemented
    │
    ▼
Validated
    │
    ▼
Deployed
    │
    ▼
Operational
    │
    ▼
Operationally Mature
```

The current Geo AI platform has reached the **Operational** stage through its Cloud Run v4 deployment, while the platform itself continues to evolve toward broader operational maturity.

---

## Quality Checks

```bash
ruff check .
ruff format --check .
pytest
pre-commit run --all-files
```

---

## License

MIT License
