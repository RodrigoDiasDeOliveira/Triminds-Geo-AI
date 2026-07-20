# src/features/adapter.py
import torch
import torch.nn as nn


class EmbeddingAdapter(nn.Module):
    """Adapter sofisticado para preservar conhecimento pré-treinado"""

    def __init__(self, in_channels: int = 64, out_channels: int = 3, bottleneck: int = 32):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


# Uso no model factory:
def create_model(config):
    if config.model.name == "resnet50":
        from torchvision.models import resnet50

        model = resnet50(weights="IMAGENET1K_V2" if config.model.pretrained else None)

        if config.model.use_adapter:
            model.conv1 = EmbeddingAdapter(
                in_channels=64, out_channels=64 if config.model.adapter_out_channels == 64 else 3
            )
        return model
