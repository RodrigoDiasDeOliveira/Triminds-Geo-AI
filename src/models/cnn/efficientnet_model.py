from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn


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