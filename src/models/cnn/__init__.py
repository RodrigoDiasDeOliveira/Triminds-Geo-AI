# src/models/cnn/__init__.py
from .efficientnet_model import build_efficientnet
from .resnet_model import build_resnet

__all__ = ["build_resnet", "build_efficientnet"]
