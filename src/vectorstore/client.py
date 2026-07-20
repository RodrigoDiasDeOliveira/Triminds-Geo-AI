# src/vectorstore/client.py
import psycopg2
from typing import List

import torch
from ..features.geo_feature import GeoFeature

class VectorStoreClient:
    """Cliente para Vector Database (pgvector, Qdrant, etc.)"""
    
    def __init__(self, connection_string: str = None):
        self.conn_str = connection_string or "postgresql://..."
        self.table = "geo_features"
    
    def upsert(self, feature: GeoFeature):
        """Insere ou atualiza feature no banco vetorial"""
        # Exemplo simplificado com pgvector
        embedding_list = feature.embedding.tolist() if feature.embedding is not None else []
        
        with psycopg2.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.table} 
                    (id, source, timestamp, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE 
                    SET embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                """, (
                    feature.feature_id,
                    feature.source,
                    feature.timestamp,
                    embedding_list,
                    feature.properties
                ))
        print(f"✅ Feature {feature.feature_id} salva no Vector Store")
    
    def similarity_search(self, query_embedding: torch.Tensor, top_k: int = 10):
        # Busca por similaridade (cosine)
        ...