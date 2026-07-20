# src/ingestion/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch

@dataclass
class GeoAsset:
    path: Path
    provider: str
    asset_type: str
    timestamp: str
    metadata: Dict
    crs: str
    resolution: float
    bbox: Optional[List] = None

@dataclass
class GeoFeature:
    geometry: Any
    timestamp: str
    embedding: Optional[torch.Tensor] = None
    properties: Dict = field(default_factory=dict)
    source: str = ""
    confidence: float = 1.0
    feature_id: Optional[str] = None

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