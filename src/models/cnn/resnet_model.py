from torchvision.models import resnet50
import torch.nn as nn


def build_resnet(num_classes: int):
    model = resnet50(weights="DEFAULT")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
