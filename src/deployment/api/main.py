import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from src.models.model_factory import build_model
from src.data_loader.dataset import default_transforms

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model("resnet50", num_classes=10)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

transform = default_transforms()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(tensor)

        prediction = torch.argmax(outputs, dim=1).item()

    return {"prediction": prediction}