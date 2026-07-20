"""
Data loading and preprocessing module for satellite imagery.

This module provides dataset classes, transformations, and data loaders
for land classification tasks.

NOTE:
This package is maintained for backward compatibility.
The long-term plan is to migrate its functionality to the
`src.ingestion` package as the Triminds platform evolves into a
multi-provider geospatial AI framework.
"""

from .dataset import (
    SatelliteDataset,
    default_transforms,
    get_dataloader,
)

# ---------------------------------------------------------------------
# Backward compatibility aliases
# TODO: Remove these aliases in v0.3.0
# ---------------------------------------------------------------------

SatelliteDataLoader = SatelliteDataset
get_transforms = default_transforms

__all__ = [
    "SatelliteDataset",
    "SatelliteDataLoader",
    "default_transforms",
    "get_transforms",
    "get_dataloader",
]
