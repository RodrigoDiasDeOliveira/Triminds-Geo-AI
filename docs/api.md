

# 🌐 API Documentation — Satellite Land Classification

## 📌 Overview

This API exposes satellite image classification models via REST interface using FastAPI.

---

## 🚀 Base URL
http://localhost:8000


---

## 📤 Endpoints

---

## 🔹 POST /predict

Classifies a satellite image.

### Request

- Content-Type: multipart/form-data

### Parameters

| Field | Type | Description |
|------|------|-------------|
| file | image | Satellite image file (RGB) |

---

### Response

```json
{
  "prediction": 3
}
Example
curl -X POST "http://localhost:8000/predict" \
-F "file=@image.jpg"
🔹 Health Check
GET /health
{
  "status": "ok"
}
🧠 Model Information
Default model: ResNet50
Input size: 224x224
Classes: 10 land cover types

🧩 Future API Extensions
/predict_batch → batch inference
/explain → Grad-CAM visualization
/drift → monitoring endpoint
/model/version → registry access
/metrics → Prometheus integration

🔐 Authentication (Future)
API Key-based auth
OAuth2 integration for enterprise deployment