from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn


def build_resnet(
    num_classes: int,
    pretrained: bool = True,
    **kwargs,
):
    weights = ResNet50_Weights.DEFAULT if pretrained else None

    model = resnet50(weights=weights, **kwargs)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model