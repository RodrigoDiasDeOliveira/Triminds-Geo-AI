 System Architecture Diagrams

---

# 🧠 1. FULL SYSTEM ARCHITECTURE

```text
[ Satellite Images ]
        ↓
[ Data Layer ]
        ↓
[ Preprocessing Pipeline ]
        ↓
[ Dataset Loader ]
        ↓
[ Model Training (CNN / ViT / Hybrid) ]
        ↓
[ MLflow Tracking ]
        ↓
[ Model Registry ]
        ↓
[ Vertex AI Deployment ]
        ↓
[ Monitoring Layer ]
        ↓
[ Auto Retraining System ]
🔄 2. MLOps LIFECYCLE
Experiment → Train → Track → Register → Deploy → Monitor → Retrain
☁️ 3. GOOGLE CLOUD ARCHITECTURE
                 Google Cloud Platform
------------------------------------------------

Cloud Storage (Satellite Data)
        ↓
Dataproc (Spark Processing)
        ↓
Vertex AI Training Jobs
        ↓
Vertex AI Endpoint (Inference)
        ↓
BigQuery GIS (Analytics)
        ↓
Monitoring System
🧠 4. MODEL PIPELINE
Input Image (224x224)
        ↓
CNN Backbone / Vision Transformer
        ↓
Feature Embedding
        ↓
Classifier Head
        ↓
Softmax Output
📡 5. MONITORING FLOW
Predictions
     ↓
Performance Monitor
     ↓
Data Drift Detector
     ↓
Model Drift Detector
     ↓
Drift Aggregator
     ↓
Alert / Retraining Trigger
     ↓
Dataproc + Vertex AI Pipeline
🚀 6. AUTO-RETRAINING LOOP
Drift Detected
      ↓
Trigger Pipeline
      ↓
Dataproc Spark Job
      ↓
Retraining
      ↓
MLflow Logging
      ↓
Model Registry Update
      ↓
Vertex AI Deployment Update