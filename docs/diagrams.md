

```markdown
# 📊 System Diagrams — Satellite Land Classification

---

## 🧠 1. Full System Architecture

```text
[ Satellite Images ]
         ↓
[ Data Ingestion Layer ]
         ↓
[ Preprocessing Pipeline ]
         ↓
[ Feature Engineering ]
         ↓
[ CNN / ViT / Hybrid Models ]
         ↓
[ Training Engine ]
         ↓
[ MLflow Tracking ]
         ↓
[ Model Registry ]
         ↓
[ Deployment Layer (FastAPI / Vertex AI) ]
         ↓
[ Monitoring Layer ]
         ↓
[ Auto Retraining Pipeline ]



 2. MLOps Lifecycle
Experiment → Train → Track → Register → Deploy → Monitor → Retrain


 3. Cloud Architecture (GCP)
                 Google Cloud Platform
-----------------------------------------------------

Cloud Storage (Dataset)
        ↓
Dataproc (Spark Processing)
        ↓
Vertex AI Training Jobs
        ↓
Vertex AI Endpoint (Serving)
        ↓
BigQuery GIS (Analytics)
        ↓
Monitoring (Drift + Metrics)


4. Model Architecture Flow
Input Image (224x224)
        ↓
CNN Backbone (ResNet / EfficientNet)
        ↓
Feature Vector (2048)
        ↓
Transformer Encoder (Optional)
        ↓
Classifier Head
        ↓
Softmax Output (10 Classes)


 5. Monitoring Flow
Prediction Output
        ↓
Performance Monitor
        ↓
Data Drift Detector
        ↓
Model Drift Detector
        ↓
Drift Monitor Aggregator
        ↓
Alert / Retraining Trigger
        ↓
Dataproc Job / Vertex Pipeline


 6. Auto-Retraining Loop
Drift Detected
      ↓
Trigger Pipeline
      ↓
Dataproc Spark Processing
      ↓
Retraining Model
      ↓
MLflow Logging
      ↓
Model Registry Update
      ↓
Deployment Update