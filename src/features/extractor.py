# src/features/extractors.py
import torch

from ..ingestion.base import GeoAsset
from .geo_feature import GeoFeature


def embedding_to_feature(asset: GeoAsset) -> GeoFeature:
    """Converte embedding bruto em GeoFeature"""
    # Carrega tensor
    # ... (usando rasterio ou torch)
    embedding = torch.rand(64)  # placeholder

    return GeoFeature(
        geometry=None,  # preencher com bbox
        timestamp=asset.timestamp,
        embedding=embedding,
        source=asset.provider,
        properties=asset.metadata,
    )
