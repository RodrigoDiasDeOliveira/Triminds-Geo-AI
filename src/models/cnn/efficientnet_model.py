from torchvision.models import efficientnet_b0
import torch.nn as nn


def build_efficientnet(num_classes):

    model = efficientnet_b0(weights="DEFAULT")

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model