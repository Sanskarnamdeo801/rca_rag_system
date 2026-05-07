"""Simple semantic retrieval for incidents."""

from __future__ import annotations

from app.utils.preprocessing import normalize_text


class RetrievalService:
    """Embed and search incident history."""

    def __init__(self, embedding_service, vector_store) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def find_similar(self, query: str, service_name: str | None = None, top_k: int = 5):
        normalized = normalize_text(query)
        query_vector = self.embedding_service.embed_text(normalized)
        return self.vector_store.search(query_vector, top_k=top_k, service_name=service_name)
