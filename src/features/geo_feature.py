# src/features/geo_feature.py
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import torch
from datetime import datetime

@dataclass
class GeoFeature:
    """Representação unificada de qualquer observação geoespacial"""
    geometry: Any                    # shapely geometry ou bbox
    timestamp: str
    embedding: Optional[torch.Tensor] = None
    properties: Dict = field(default_factory=dict)
    source: str = ""
    confidence: float = 1.0
    feature_id: Optional[str] = None
    tags: torch.List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.feature_id is None:
            self.feature_id = f"{self.source}_{self.timestamp}"
    
    def to_dict(self) -> Dict:
        return {
            "id": self.feature_id,
            "geometry": str(self.geometry),
            "timestamp": self.timestamp,
            "source": self.source,
            "embedding_dim": len(self.embedding) if self.embedding is not None else 0,
            "confidence": self.confidence,
            **self.properties
        }