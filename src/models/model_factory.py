from src.models.cnn.resnet_model import build_resnet
from src.models.cnn.efficientnet_model import build_efficientnet

from src.models.vit.vit_model import build_vit
from src.models.vit.swin_transformer import (
    build_swin_transformer
)


def build_model(model_name, num_classes):

    if model_name == "resnet50":
        return build_resnet(num_classes)

    if model_name == "efficientnet":
        return build_efficientnet(num_classes)

    if model_name == "vit":
        return build_vit(num_classes)

    if model_name == "swin":
        return build_swin_transformer(num_classes)

    raise ValueError(f"Unsupported model: {model_name}")