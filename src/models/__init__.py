# src/models/__init__.py
from .cnn.efficientnet_model import build_efficientnet

# Exportar modelos principais
from .cnn.resnet_model import build_resnet
from .hybrid.cnn_transformer_hybrid import CNNTransformerHybrid
from .model_factory import build_model
from .vit.swin_transformer import build_swin_transformer
from .vit.vit_model import build_vit

__all__ = [
    "build_model",
    "build_resnet",
    "build_efficientnet",
    "build_vit",
    "build_swin_transformer",
    "CNNTransformerHybrid",
]