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

```bash
satellite-land-classification/
├── data/                    # Raw and processed datasets
├── notebooks/               # Exploratory analysis and experiments
├── src/
│   ├── data_loader/         # Data loading and preprocessing
│   ├── models/              # Model architectures (CNN, ViT, Hybrid)
│   ├── training/            # Training scripts and configurations
│   ├── evaluation/          # Metrics and inference
│   └── utils/               # Helper functions
├── gcp/                     # GCP and Spark configurations
│   ├── dataproc/
│   └── vertex_pipelines/
├── config/                  # YAML configuration files
├── experiments/             # Experiment tracking
├── requirements.txt
├── Dockerfile
└── README.md
