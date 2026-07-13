# src/models/cnn/__init__.py
from .resnet_model import build_resnet
from .efficientnet_model import build_efficientnet

__all__ = ["build_resnet", "build_efficientnet"]