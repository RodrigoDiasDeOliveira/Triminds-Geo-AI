import io
import os
from functools import lru_cache

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.data_loader.dataset import default_transforms
from src.models.model_factory import build_model

app = FastAPI(title="Satellite Land Classification API")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pth")
MODEL_NAME = os.environ.get("MODEL_NAME", "resnet50")
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "10"))


@lru_cache(maxsize=1)
def get_model():
    model = build_model(MODEL_NAME, num_classes=NUM_CLASSES)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    model.to(DEVICE).eval()
    return model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": os.path.exists(MODEL_PATH),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as err:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {err}",
        ) from err

    tensor = default_transforms()(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = get_model()(tensor)
        prediction = int(torch.argmax(outputs, dim=1).item())

    return {"prediction": prediction}