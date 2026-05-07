"""Embedding tests for the MVP."""

from __future__ import annotations

import numpy as np

from app.config import Settings
from app.services.embedding_service import EmbeddingService


def test_hashing_embeddings_are_fixed_dimension(settings: Settings) -> None:
    service = EmbeddingService(settings)
    vector = service.embed_text("redis connection reset by peer")

    assert vector.shape == (settings.embedding_dimension,)
    assert np.linalg.norm(vector) > 0
