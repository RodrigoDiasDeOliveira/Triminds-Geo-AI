# src/ingestion/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class GeoAsset:
    path: Path
    provider: str
    asset_type: str
    timestamp: str
    metadata: dict
    crs: str
    resolution: float
    bbox: list | None = None


@dataclass
class GeoFeature:
    geometry: Any
    timestamp: str
    embedding: torch.Tensor | None = None
    properties: dict = field(default_factory=dict)
    source: str = ""
    confidence: float = 1.0
    feature_id: str | None = None


class GeoProvider(ABC):
    name: str
    provider_type: str

    @abstractmethod
    def export(self, **kwargs):
        pass

    @abstractmethod
    def load_asset(self, asset_path: Path) -> GeoAsset:
        pass

    @abstractmethod
    def extract_features(self, asset: GeoAsset) -> GeoFeature:
        pass
