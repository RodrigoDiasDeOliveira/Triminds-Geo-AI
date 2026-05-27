# 🌐 API — Satellite Land Classification

## 📌 Overview

REST API for satellite image classification using deep learning models.

Built with FastAPI and integrated with MLflow-trained models.

---

# 🚀 Base URL


http://localhost:8000


---

# 📤 ENDPOINTS

---

## 🔹 POST /predict

Classifies a satellite image.

### Request

- Content-Type: multipart/form-data

| Field | Type | Description |
|------|------|-------------|
| file | image | Satellite image (RGB) |

---

### Response

```json
{
  "prediction": 2
}
🔹 GET /health

Returns API health status.

{
  "status": "ok"
}
🔹 Future Endpoints
/predict_batch → batch inference
/explain → model explainability (Grad-CAM)
/drift → monitoring integration
/metrics → performance stats
/model/version → registry info

🧠 MODEL DETAILS
Architecture: ResNet50 / ViT / Hybrid CNN-Transformer
Input size: 224x224
Output: Multi-class land classification

🔐 AUTH (FUTURE)
API Key authentication
OAuth2 for enterprise deployments