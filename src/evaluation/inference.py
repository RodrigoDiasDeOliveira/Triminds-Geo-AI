import torch
from PIL import Image

from src.data_loader.dataset import default_transforms


def predict(model, image_path, device):

    model.eval()

    image = Image.open(image_path).convert("RGB")

    transform = default_transforms()

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)

        prediction = torch.argmax(outputs, dim=1)

    return prediction.item()
