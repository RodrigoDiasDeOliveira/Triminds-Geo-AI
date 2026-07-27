# src/vectorstore/client.py
import logging

import psycopg2
import torch
from psycopg2 import sql

from ..features.geo_feature import GeoFeature

logger = logging.getLogger(__name__)


class VectorStoreClient:
    """Client for vector databases such as pgvector."""

    def __init__(
        self,
        connection_string: str | None = None,
    ) -> None:
        self.conn_str = connection_string or "postgresql://..."
        self.table = "geo_features"

    def upsert(
        self,
        feature: GeoFeature,
    ) -> None:
        """Insert or update a feature in the vector store."""

        if feature.embedding is None:
            embedding_list: list[float] = []
        else:
            embedding_list = feature.embedding.tolist()

        query = sql.SQL(
            """
            INSERT INTO {}
            (id, source, timestamp, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
            """
        ).format(sql.Identifier(self.table))

        with psycopg2.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        feature.feature_id,
                        feature.source,
                        feature.timestamp,
                        embedding_list,
                        feature.properties,
                    ),
                )

        logger.info(
            "Feature %s saved to Vector Store.",
            feature.feature_id,
        )

    def similarity_search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 10,
    ) -> list[GeoFeature]:
        """
        Search for the most similar vectors.

        This method must be implemented by the concrete vector store.
        """
        raise NotImplementedError
