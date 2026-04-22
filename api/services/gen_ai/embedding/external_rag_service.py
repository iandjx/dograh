"""External RAG service for querying a user-provided PostgreSQL table."""

from typing import Any, Dict, List, Optional

import asyncpg
from loguru import logger
from openai import AsyncOpenAI

from .base import BaseEmbeddingService

DEFAULT_MODEL_ID = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


class ExternalRAGService(BaseEmbeddingService):
    """Queries a user-supplied PostgreSQL table for RAG retrieval.

    Generates query embeddings via OpenAI (or compatible endpoint), then runs
    a pgvector cosine similarity search against the configured table. Requires
    the pgvector extension on the target database.

    Designed for tables like n8n_vectors:
        id uuid, text text, metadata jsonb, embedding vector

    Column defaults (override via EXTERNAL_RAG_*_COLUMN env vars):
      text:      EXTERNAL_RAG_TEXT_COLUMN      (default: "text")
      embedding: EXTERNAL_RAG_EMBEDDING_COLUMN (default: "embedding")
      name:      EXTERNAL_RAG_NAME_COLUMN      (default: "" → metadata->>'source')

    When EXTERNAL_RAG_NAME_COLUMN is empty the service always fetches the
    'metadata' column and extracts metadata->>'source' as the display name,
    falling back to the table name when not present.
    """

    def __init__(
        self,
        db_url: str,
        table: str,
        api_key: str,
        model_id: str = DEFAULT_MODEL_ID,
        base_url: Optional[str] = None,
        text_column: str = "text",
        embedding_column: str = "embedding",
        name_column: str = "",
    ):
        self.db_url = db_url
        self.table = table
        self.text_column = text_column
        self.embedding_column = embedding_column
        # Empty string means "derive from metadata->>'source'"
        self.name_column = name_column
        self.model_id = model_id

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)

    def get_model_id(self) -> str:
        return self.model_id

    def get_embedding_dimension(self) -> int:
        return EMBEDDING_DIMENSION

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(input=texts, model=self.model_id)
        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> List[float]:
        return (await self.embed_texts([query]))[0]

    async def search_similar_chunks(
        self,
        query: str,
        organization_id: int,
        limit: int = 5,
        document_uuids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run pgvector cosine similarity search against the external table.

        organization_id and document_uuids are ignored — the external table is
        assumed to already be scoped to the right data.
        """
        query_embedding = await self.embed_query(query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        use_metadata_for_name = not self.name_column
        name_select = (
            "metadata" if use_metadata_for_name else f"{self.name_column} AS name"
        )

        sql = f"""
            SELECT
                {self.text_column}   AS chunk_text,
                {name_select},
                1 - ({self.embedding_column} <=> $1::vector) AS similarity
            FROM {self.table}
            ORDER BY {self.embedding_column} <=> $1::vector
            LIMIT $2
        """

        conn = await asyncpg.connect(self.db_url)
        try:
            rows = await conn.fetch(sql, embedding_str, limit)
            logger.debug(
                f"External RAG query on {self.table}: {len(rows)} results returned"
            )
        finally:
            await conn.close()

        results = []
        for i, row in enumerate(rows):
            if use_metadata_for_name:
                meta = row["metadata"] or {}
                filename = meta.get("source", self.table)
            else:
                filename = row["name"]

            results.append(
                {
                    "chunk_text": row["chunk_text"],
                    "contextualized_text": None,
                    "filename": filename,
                    "metadata": row["metadata"] if use_metadata_for_name else None,
                    "similarity": float(row["similarity"]),
                    "chunk_index": i,
                }
            )

        return results
