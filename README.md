# 🛰️ Satellite Land Classification Platform

A production-grade **cloud-native machine learning platform** for satellite image classification focused on:

- Land use classification
- Precision agriculture
- Geospatial intelligence
- Scalable ML pipelines on Google Cloud

Built with PyTorch, MLOps practices, and GCP services (Vertex AI, Dataproc, BigQuery).

---

# 🚀 Project Goals

- Build high-performance land cover classification models
- Compare CNNs, Vision Transformers, and Hybrid architectures
- Create scalable geospatial ML pipelines
- Enable production deployment on Google Cloud Platform
- Implement full MLOps lifecycle (train → track → deploy → monitor → retrain)

---

# 🧠 Core Capabilities

## 🔬 Machine Learning
- CNN models (ResNet, EfficientNet)
- Vision Transformers (ViT, Swin Transformer)
- Hybrid CNN + Transformer architecture
- Custom model factory system
- Training pipeline with PyTorch

## 🌍 Geospatial Intelligence
- Satellite image preprocessing
- Geo-aware utilities
- Raster and coordinate handling
- Ready for multi-spectral extension (Sentinel-2)

## ⚙️ MLOps System
- MLflow experiment tracking
- Model registry system
- Experiment tracking module
- Structured logging system
- Dataproc orchestration jobs

## 📊 Monitoring System
- Data drift detection (KS test)
- Model drift detection
- Performance monitoring
- Drift aggregation system
- Auto-retraining trigger design

## ☁️ Cloud Infrastructure (GCP)

Integrated with Google Cloud Platform:

- Vertex AI (training + deployment)
- Dataproc (Spark distributed processing)
- Cloud Storage (datasets)
- BigQuery GIS (geospatial analytics)
- Terraform (infra-as-code)

---

# 🏗️ Architecture Overview

## System Layers

```text
DATA LAYER
  ├── data_loader
  ├── preprocessing

MODEL LAYER
  ├── CNN / ViT / Hybrid models

PIPELINE LAYER
  ├── training_pipeline
  ├── preprocessing_pipeline
  ├── vertex_pipeline

MLOps LAYER
  ├── mlops/tracking
  ├── mlops/registry
  ├── mlops/orchestration
  ├── mlops/logging

MONITORING LAYER
  ├── data_drift
  ├── model_drift
  ├── performance_monitor

CLOUD LAYER
  ├── gcp/dataproc
  ├── gcp/vertex_ai
  ├── gcp/bigquery
  ├── gcp/terraform
``` id="arch1"

---

# 🔄 System Workflow

```text id="flow1"
Satellite Images
      ↓
Data Ingestion (GeoTIFF / Raster)
      ↓
Preprocessing Pipeline
      ↓
Dataset Loader
      ↓
Model Training (CNN / ViT / Hybrid)
      ↓
MLflow Tracking
      ↓
Model Registry
      ↓
Vertex AI Deployment
      ↓
Monitoring Layer
      ↓
Auto-Retraining (Dataproc + Vertex AI Pipelines)
🌐 API

FastAPI-based inference service.

Endpoint
POST /predict

Upload a satellite image and receive a classification.

curl -X POST "http://localhost:8000/predict" \
-F "file=@image.jpg"
Response
{
  "prediction": 2
}
Health Check
{
  "status": "ok"
}
🧪 Testing Strategy

The project includes 3 levels of testing:

Unit Tests
Models
Metrics
Dataset
Geospatial utilities
MLOps components
Integration Tests
Training pipeline
Vertex AI pipeline
Dataproc jobs
FastAPI endpoints
Performance Tests
Inference latency
Memory usage
Batch throughput
🧪 Run Tests
pytest tests/
🐳 Docker
docker build -t satellite-classification .

docker run -p 8000:8000 satellite-classification
🏋️ Training
python src/pipelines/training_pipeline.py
☁️ Cloud Stack
Google Cloud Vertex AI
Google Cloud Dataproc
Google Cloud Storage
BigQuery GIS
Terraform Infrastructure
📈 Supported Models
ResNet50
EfficientNet
Vision Transformer (ViT)
Swin Transformer
CNN + Transformer Hybrid
🔬 Future Enhancements
ML
Semantic segmentation (UNet, DeepLabV3)
Self-supervised learning
Multi-spectral satellite support
Geospatial
Sentinel-2 ingestion pipeline
Temporal satellite analysis
Raster-based training pipeline
MLOps
Full CI/CD pipeline
Shadow deployment testing
Automated retraining system
Cloud
Kubernetes (GKE) deployment
Streaming ingestion (Pub/Sub)
Real-time inference monitoring
🧠 Design Principles
Modular architecture
Separation of concerns (ML / MLOps / Monitoring / Infra)
Cloud-native design
Reproducibility
Production-first mindset
Scalable pipeline architecture
📊 Project Status

🟢 Active Development
Phase: MLOps + Cloud Integration Complete
Next: Production deployment & advanced geospatial pipelines

📜 License

MIT License
