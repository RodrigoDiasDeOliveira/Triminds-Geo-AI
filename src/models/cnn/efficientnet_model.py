import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet(
    num_classes: int,
    pretrained: bool = True,
    **kwargs,
):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None

    model = efficientnet_b0(
        weights=weights,
        **kwargs,
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes,
    )

    return model