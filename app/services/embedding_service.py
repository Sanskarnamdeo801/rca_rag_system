"""Embedding generation with a singleton sentence-transformer and hashing fallback."""

from __future__ import annotations

import hashlib
import threading
from functools import lru_cache

import numpy as np

from app.config import Settings
from app.utils.logger import get_logger

LOGGER = get_logger(__name__)

try:  # pragma: no cover - import availability depends on runtime environment
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - handled by fallback at runtime
    SentenceTransformer = None  # type: ignore[assignment]


class HashingBackend:
    """Deterministic lightweight fallback when the model is unavailable."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = text.split()
            if not tokens:
                continue
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, index] += sign
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors


class EmbeddingService:
    """Generate embeddings using a shared model instance."""

    _model = None
    _model_name: str | None = None
    _model_lock = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._hashing_backend = HashingBackend(settings.embedding_dimension)

    @property
    def backend_name(self) -> str:
        if self.settings.embedding_backend == "hashing":
            return "hashing"
        if SentenceTransformer is None:
            return "hashing"
        return "sentence-transformers" if self._model is not None else "warming"

    def warmup(self) -> None:
        """Warm the shared model once without failing startup."""

        if self.settings.embedding_backend == "hashing":
            LOGGER.info("Embedding backend set to hashing.")
            return
        try:
            self._get_model()
            LOGGER.info("Embedding model loaded.")
        except Exception as exc:  # pragma: no cover - depends on local model state
            LOGGER.error("Embedding warmup failed. Falling back to hashing. error=%s", exc)

    def _get_model(self):
        if self.settings.embedding_backend == "hashing":
            return None
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not available")
        if EmbeddingService._model is None or EmbeddingService._model_name != self.settings.embedding_model_name:
            with EmbeddingService._model_lock:
                if EmbeddingService._model is None or EmbeddingService._model_name != self.settings.embedding_model_name:
                    EmbeddingService._model = SentenceTransformer(self.settings.embedding_model_name)
                    EmbeddingService._model_name = self.settings.embedding_model_name
        return EmbeddingService._model

    @lru_cache(maxsize=1024)
    def _embed_single_cached(self, text: str) -> tuple[float, ...]:
        vectors = self.embed_texts([text])
        return tuple(float(value) for value in vectors[0])

    def embed_text(self, text: str) -> np.ndarray:
        return np.asarray(self._embed_single_cached(text), dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed one or more texts, always returning normalized vectors."""

        if not texts:
            return np.zeros((0, self.settings.embedding_dimension), dtype=np.float32)

        normalized = [text.strip() for text in texts]

        if self.settings.embedding_backend == "hashing":
            return self._hashing_backend.encode(normalized)

        try:
            model = self._get_model()
            if model is None:
                return self._hashing_backend.encode(normalized)
            vectors = model.encode(
                normalized,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return np.asarray(vectors, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            LOGGER.error("Embedding generation failed. Falling back to hashing. error=%s", exc)
            return self._hashing_backend.encode(normalized)
