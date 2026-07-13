# src/models/model_factory.py
"""
Model Factory - Centralized model creation
Supports: ResNet, EfficientNet, ViT, Swin Transformer, Hybrid
"""

from src.models.cnn.resnet_model import build_resnet
from src.models.cnn.efficientnet_model import build_efficientnet
from src.models.vit.vit_model import build_vit
from src.models.vit.swin_transformer import build_swin_transformer
from src.models.hybrid.cnn_transformer_hybrid import CNNTransformerHybrid


def build_model(model_name: str, num_classes: int = 10, pretrained: bool = True, **kwargs):
    """
    Factory function to build different models.
    """
    name = model_name.lower().strip()

    if name in ("resnet50", "resnet"):
        return build_resnet(num_classes=num_classes, pretrained=pretrained, **kwargs)
    
    elif name in ("efficientnet", "efficientnet_b0"):
        return build_efficientnet(num_classes=num_classes, pretrained=pretrained, **kwargs)
    
    elif name in ("vit", "vit_base_patch16_224"):
        return build_vit(num_classes=num_classes, pretrained=pretrained, **kwargs)
    
    elif name in ("swin", "swin_base_patch4_window7_224"):
        return build_swin_transformer(num_classes=num_classes, pretrained=pretrained, **kwargs)
    
    elif name == "hybrid":
        return CNNTransformerHybrid(num_classes=num_classes, **kwargs)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}. "
                        f"Available: resnet, efficientnet, vit, swin, hybrid")