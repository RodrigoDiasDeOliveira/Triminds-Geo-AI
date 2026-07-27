# src/workflows/ingestion_pipeline.py

# src/workflows/ingestion_pipeline.py

from ..features import engine
from ..vectorstore import client


class IngestionPipeline:
    """Pipeline completo de ingestão."""

    def __init__(
        self,
        engine_instance: engine.GeoFeatureEngine | None = None,
        vector_store: client.VectorStoreClient | None = None,
    ) -> None:
        self.engine = engine_instance or engine.GeoFeatureEngine()
        self.vector_store = vector_store or client.VectorStoreClient()

    def run(self, config: dict):
        features = self.engine.ingest_and_extract(config)

        for feature in features:
            self.vector_store.upsert(feature)

        print(f"✅ Ingestão concluída: {len(features)} features processadas")

        return features
