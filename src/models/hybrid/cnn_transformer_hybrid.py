import torch
import torch.nn as nn
import torchvision.models as models


class CNNTransformerHybrid(nn.Module):

    def __init__(self, num_classes=10):

        super(CNNTransformerHybrid, self).__init__()

        # Backbone CNN (ResNet)
        self.cnn = models.resnet50(weights="DEFAULT")
        self.cnn.fc = nn.Identity()

        cnn_out = 2048

        # Projection layer para Transformer
        self.projection = nn.Linear(cnn_out, 512)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # Classifier final
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        # CNN features
        x = self.cnn(x)              # (B, 2048)

        x = self.projection(x)       # (B, 512)

        # Transformer espera sequência → reshape fake sequence
        x = x.unsqueeze(1)           # (B, 1, 512)

        x = self.transformer(x)

        x = x.squeeze(1)

        x = self.classifier(x)

        return x