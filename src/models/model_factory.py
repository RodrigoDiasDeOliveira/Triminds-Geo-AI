from src.models.cnn.resnet_model import build_resnet
from src.models.cnn.efficientnet_model import build_efficientnet
from src.models.vit.vit_model import build_vit
from src.models.vit.swin_transformer import build_swin_transformer
from src.models.hybrid.cnn_transformer_hybrid import CNNTransformerHybrid


def build_model(model_name: str, num_classes: int):
    name = model_name.lower()
    if name in ("resnet50", "resnet"):
        return build_resnet(num_classes)
    if name in ("efficientnet", "efficientnet_b0"):
        return build_efficientnet(num_classes)
    if name in ("vit", "vit_base_patch16_224"):
        return build_vit(num_classes)
    if name in ("swin", "swin_base_patch4_window7_224"):
        return build_swin_transformer(num_classes)
    if name == "hybrid":
        return CNNTransformerHybrid(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")
