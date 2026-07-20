# src/workflows/ingestion_pipeline.py

from ..features.engine import GeoFeatureEngine
from ..vectorstore.client import VectorStoreClient


class IngestionPipeline:
    """Pipeline completo de ingestão"""

    def __init__(self):
        self.engine = GeoFeatureEngine()
        self.vector_store = VectorStoreClient()

    def run(self, config: dict):
        features = self.engine.ingest_and_extract(config)

        # Salva no Vector Store
        for feature in features:
            self.vector_store.upsert(feature)

        print(f"✅ Ingestão concluída: {len(features)} features processadas")
        return features
