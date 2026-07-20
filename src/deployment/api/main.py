import io
import os
from functools import lru_cache
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.data_loader.dataset import default_transforms
from src.models.model_factory import build_model

load_dotenv()

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/config.yaml")


def _load_config() -> dict:
    p = Path(CONFIG_PATH)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


CONFIG = _load_config()
MODEL_CFG = CONFIG.get("model", {})
DATA_CFG = CONFIG.get("data", {})
DATASET_CFG = CONFIG.get("dataset", {})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.pth")
MODEL_NAME = os.environ.get("MODEL_NAME", MODEL_CFG.get("name", "resnet50"))
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", DATA_CFG.get("num_classes", 10)))
IN_CHANNELS = int(os.environ.get("IN_CHANNELS", MODEL_CFG.get("in_channels", 3)))
USE_ADAPTER = os.environ.get("USE_ADAPTER", str(MODEL_CFG.get("use_adapter", False))).lower() == "true"
ADAPTER_OUT = int(MODEL_CFG.get("adapter_out_channels", 64))
CLASS_NAMES: list[str] = DATASET_CFG.get("classes", []) or [str(i) for i in range(NUM_CLASSES)]

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://localhost:4173",
    ).split(",")
    if o.strip()
]

app = FastAPI(title="Triminds Geo AI - Satellite Land Classification API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_model():
    model = build_model(
        MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=False,
        in_channels=IN_CHANNELS,
        use_adapter=USE_ADAPTER,
        adapter_out_channels=ADAPTER_OUT,
    )
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_name": MODEL_NAME,
        "model_loaded": os.path.exists(MODEL_PATH),
        "num_classes": NUM_CLASSES,
    }


@app.get("/classes")
def classes():
    return {"classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Invalid image: {err}") from err

    tensor = default_transforms()(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = get_model()(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    class_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
    return {"prediction": pred_idx, "class_name": class_name, "confidence": confidence}
