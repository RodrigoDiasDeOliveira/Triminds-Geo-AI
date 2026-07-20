"""
Data loading and preprocessing module for satellite imagery.

This module provides dataset classes, transformations, and data loaders
for land classification tasks.
"""

# Para manter compatibilidade com o que já estava no __init__.py anterior
from .dataset import SatelliteDataset
from .dataset import SatelliteDataset as SatelliteDataLoader
from .dataset import default_transforms
from .dataset import default_transforms as get_transforms
from .dataset import get_dataloader

__all__ = [
    "SatelliteDataset",
    "SatelliteDataLoader",  # alias para compatibilidade
    "get_dataloader",
    "default_transforms",
    "get_transforms",  # alias para compatibilidade
]
