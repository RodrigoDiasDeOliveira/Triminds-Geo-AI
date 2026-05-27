# Satellite Land Classification

A deep learning project for **land use and crop classification** using high-resolution satellite imagery.

This project leverages modern computer vision techniques to identify different types of crops and land covers, with a strong focus on **precision agriculture** and **remote sensing**.

### Key Objectives
- Build high-accuracy models for multi-class land cover classification
- Experiment with CNNs, Vision Transformers (ViT), and hybrid architectures
- Develop scalable data pipelines for large satellite datasets
- Prepare the solution for production using Google Cloud Platform and Apache Spark

---

### Technologies

**Core Stack:**
- **PyTorch** & **Torchvision**
- **Vision Transformers (ViT)**
- **CNNs** and Hybrid Models (CNN + ViT)
- **Albumentations** / **Torchvision** for data augmentation

**Planned / Scalability:**
- **Apache Spark** (PySpark) for distributed data processing
- **Google Cloud Platform**:
  - Google Cloud Storage (GCS)
  - Dataproc (Spark clusters)
  - Vertex AI (training and deployment)
  - BigQuery (metadata and analytics)

**Tools:**
- Python 3.10+
- Jupyter Notebooks
- Docker

---

### Project Structure

satellite-land-classification/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── interim/
│   ├── external/
│   └── samples/
│
├── notebooks/
│   ├── exploratory/
│   ├── feature_engineering/
│   └── experiments/
│
├── src/
│   ├── data_loader/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── augmentations.py
│   │   └── geospatial_utils.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn/
│   │   │   ├── __init__.py
│   │   │   ├── resnet_model.py
│   │   │   └── efficientnet_model.py
│   │   │
│   │   ├── vit/
│   │   │   ├── __init__.py
│   │   │   ├── vit_model.py
│   │   │   └── swin_transformer.py
│   │   │
│   │   ├── hybrid/
│   │   │   ├── __init__.py
│   │   │   └── cnn_transformer_hybrid.py
│   │   │
│   │   └── model_factory.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── distributed_training.py
│   │   ├── callbacks.py
│   │   └── losses.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── confusion_matrix.py
│   │   ├── inference.py
│   │   └── explainability.py
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── preprocessing_pipeline.py
│   │   ├── training_pipeline.py
│   │   └── vertex_pipeline.py
│   │
│   ├── deployment/
│   │   ├── api/
│   │   │   └── main.py
│   │   └── vertex_endpoint/
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── config_loader.py
│       ├── seed.py
│       └── storage.py
│
├── config/
│   ├── dataset.yaml
│   ├── model.yaml
│   ├── training.yaml
│   └── gcp.yaml
│
├── gcp/
│   ├── dataproc/
│   ├── vertex_ai/
│   ├── terraform/
│   └── bigquery/
│
├── experiments/
│   ├── mlflow/
│   └── tensorboard/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
│
├── scripts/
│   ├── train.sh
│   ├── evaluate.sh
│   └── deploy.sh
│
├── docs/
│   ├── architecture/
│   ├── diagrams/
│   └── api/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
