"""
Model Factory - Centralized model creation
Supports: ResNet, EfficientNet, ViT, Swin Transformer, Hybrid + Embedding Adapters
"""

# Import do novo adapter
from src.features.adapter import get_adapter
from src.models.cnn.efficientnet_model import build_efficientnet
from src.models.cnn.resnet_model import build_resnet
from src.models.hybrid.cnn_transformer_hybrid import CNNTransformerHybrid
from src.models.vit.swin_transformer import build_swin_transformer
from src.models.vit.vit_model import build_vit


def build_model(
    model_name: str,
    num_classes: int = 10,
    pretrained: bool = True,
    in_channels: int = 3,
    use_adapter: bool = False,
    adapter_out_channels: int = 64,
    **kwargs
):
    """
    Factory function to build different models with support for Embedding Adapters.
    """
    name = model_name.lower().strip()

    # ====================== MODEL CREATION ======================
    if name in ("resnet50", "resnet"):
        model = build_resnet(
            num_classes=num_classes, 
            pretrained=pretrained, 
            **kwargs
        )
        first_layer_name = "conv1"

    elif name in ("efficientnet", "efficientnet_b0"):
        model = build_efficientnet(
            num_classes=num_classes, 
            pretrained=pretrained, 
            **kwargs
        )
        first_layer_name = "conv_stem"  # ou _conv_stem dependendo da implementação

    elif name in ("vit", "vit_base_patch16_224"):
        model = build_vit(
            num_classes=num_classes, 
            pretrained=pretrained, 
            **kwargs
        )
        # ViT geralmente não tem conv1 simples, precisa de patch embedding adaptation
        first_layer_name = None  # tratar separadamente se necessário

    elif name in ("swin", "swin_base_patch4_window7_224"):
        model = build_swin_transformer(
            num_classes=num_classes, 
            pretrained=pretrained, 
            **kwargs
        )
        first_layer_name = None

    elif name == "hybrid":
        model = CNNTransformerHybrid(num_classes=num_classes, **kwargs)
        first_layer_name = None

    else:
        raise ValueError(f"Unsupported model: {model_name}. "
                        f"Available: resnet, efficientnet, vit, swin, hybrid")

    # ====================== ADAPTER INTEGRATION ======================
    if use_adapter and in_channels == 64:
        print(f"🔧 Applying EmbeddingAdapter to {model_name}")
        
        adapter = get_adapter(
            adapter_type="embedding",
            in_channels=64,
            out_channels=adapter_out_channels,
            bottleneck_dim=32
        )
        
        # Aplica o adapter na primeira camada convolucional (se existir)
        if first_layer_name and hasattr(model, first_layer_name):
            setattr(model, first_layer_name, adapter)
        else:
            print(f"⚠️  Modelo {model_name} não possui camada convolucional padrão. "
                  "Adapter aplicado manualmente ou ignorado.")

    return model