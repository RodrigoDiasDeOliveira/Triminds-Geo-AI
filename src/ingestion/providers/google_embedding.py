# src/ingestion/providers/google_embedding.py
import ee
from pathlib import Path
from typing import Optional, Dict
from ..base import GeoProvider, GeoAsset, GeoFeature
from ..export_manager import ExportManager
from ...features.metadata import create_metadata
from ...features.adapter import EmbeddingAdapter

class GoogleEmbeddingProvider(GeoProvider):
    name = "google_embedding"
    provider_type = "embedding"
    collection_id = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    
    def __init__(self, bucket_name: str, project_id: Optional[str] = None):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.export_manager = ExportManager()
        ee.Initialize(project=project_id)
    
    def export(self, year: int, region: ee.Geometry, output_prefix: str = "embeddings", **kwargs):
        return self.export_manager.submit(
            collection=self.collection_id,
            year=year,
            region=region,
            bucket=self.bucket_name,
            prefix=f"{output_prefix}/{year}/"
        )
    
    def load_asset(self, asset_path: Path) -> GeoAsset:
        metadata = create_metadata(asset_path, self.name, year=2023)  # extrai do JSON se existir
        return GeoAsset(
            path=asset_path,
            provider=self.name,
            asset_type="embedding",
            timestamp=metadata["timestamp"],
            metadata=metadata,
            crs=metadata.get("crs", "EPSG:4326"),
            resolution=10.0
        )
    
    def extract_features(self, asset: GeoAsset) -> GeoFeature:
        # Carrega embedding e converte para GeoFeature
        from ...features.extractors import embedding_to_feature
        return embedding_to_feature(asset)