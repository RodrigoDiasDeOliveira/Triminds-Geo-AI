import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class CNNTransformerHybrid(nn.Module):
    """
    Hybrid CNN + Transformer model.
    Uses a ResNet50 backbone followed by a Transformer Encoder.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()

        # CNN Backbone
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.cnn = resnet50(weights=weights)
        self.cnn.fc = nn.Identity()

        cnn_out = 2048

        # Feature Projection
        self.projection = nn.Linear(cnn_out, 512)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4,
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)

        # Projection
        x = self.projection(x)

        # Add sequence dimension for Transformer
        x = x.unsqueeze(1)

        # Transformer
        x = self.transformer(x)

        # Remove sequence dimension
        x = x.squeeze(1)

        # Classification
        x = self.classifier(x)

        return x