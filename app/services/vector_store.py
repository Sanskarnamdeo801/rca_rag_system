"""Persistent FAISS-backed vector storage with a NumPy fallback."""

from __future__ import annotations

import json
import threading

import numpy as np

from app.config import Settings
from app.models import Incident, SimilarIncident
from app.utils.logger import get_logger
from app.utils.preprocessing import build_embedding_text

LOGGER = get_logger(__name__)

try:  # pragma: no cover - depends on runtime environment
    import faiss
except Exception:  # pragma: no cover - handled by NumPy fallback
    faiss = None  # type: ignore[assignment]


class VectorStore:
    """Load, persist, and search incidents using FAISS or NumPy."""

    def __init__(self, settings: Settings, embedding_service) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.dimension = settings.embedding_dimension
        self.lock = threading.RLock()
        self.incidents: list[Incident] = []
        self.embeddings = np.zeros((0, self.dimension), dtype=np.float32)
        self.index = None
        self.index_ready = False
        self.backend = "faiss" if faiss is not None else "numpy"

    @property
    def size(self) -> int:
        return len(self.incidents)

    def initialize(self) -> None:
        """Prepare persisted files and load the current dataset into memory."""

        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.faiss_dir.mkdir(parents=True, exist_ok=True)
        self._seed_incidents_if_needed()
        self._load_incidents()
        self._load_embeddings()
        LOGGER.info("Vector store initialized. incidents=%s backend=%s index_ready=%s", self.size, self.backend, self.index_ready)

    def warmup(self) -> None:
        """Build the vector index in the background when needed."""

        if self.size and not self.index_ready:
            self.rebuild_index()

    def _seed_incidents_if_needed(self) -> None:
        if self.settings.incidents_path.exists():
            try:
                existing = json.loads(self.settings.incidents_path.read_text(encoding="utf-8"))
                if isinstance(existing, list) and existing:
                    return
            except Exception:
                pass

        if self.settings.sample_incidents_path.exists():
            self.settings.incidents_path.write_text(
                self.settings.sample_incidents_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        else:
            self.settings.incidents_path.write_text("[]", encoding="utf-8")

    def _load_incidents(self) -> None:
        try:
            payload = json.loads(self.settings.incidents_path.read_text(encoding="utf-8"))
        except Exception:
            payload = []
        self.incidents = [Incident.model_validate(item) for item in payload if isinstance(item, dict)]

    def _load_embeddings(self) -> None:
        if self.settings.embeddings_path.exists():
            try:
                embeddings = np.load(self.settings.embeddings_path)
                if embeddings.ndim == 2 and embeddings.shape[0] == self.size and embeddings.shape[1] == self.dimension:
                    self.embeddings = embeddings.astype(np.float32)
                    if self.backend == "faiss" and faiss is not None and self.settings.faiss_index_path.exists():
                        self.index = faiss.read_index(str(self.settings.faiss_index_path))
                    else:
                        self._build_runtime_index()
                    self.index_ready = self.size == 0 or self.index is not None or self.backend == "numpy"
                    return
            except Exception as exc:
                LOGGER.error("Failed to load persisted embeddings. error=%s", exc)

        self.embeddings = np.zeros((0, self.dimension), dtype=np.float32)
        self.index_ready = self.size == 0
        self._build_runtime_index()

    def _build_runtime_index(self) -> None:
        if self.backend != "faiss" or faiss is None:
            self.index = None
            return
        index = faiss.IndexFlatIP(self.dimension)
        if self.embeddings.size:
            index.add(self.embeddings)
        self.index = index

    def _save_incidents(self) -> None:
        payload = [incident.model_dump(mode="json") for incident in self.incidents]
        self.settings.incidents_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_embeddings(self) -> None:
        np.save(self.settings.embeddings_path, self.embeddings)
        if self.backend == "faiss" and faiss is not None and self.index is not None:
            faiss.write_index(self.index, str(self.settings.faiss_index_path))

    def rebuild_index(self) -> None:
        """Recompute all embeddings and rebuild the persistent index."""

        with self.lock:
            if not self.incidents:
                self.embeddings = np.zeros((0, self.dimension), dtype=np.float32)
                self._build_runtime_index()
                self.index_ready = True
                self._save_incidents()
                self._save_embeddings()
                return

            texts = [build_embedding_text(incident) for incident in self.incidents]
            self.embeddings = self.embedding_service.embed_texts(texts)
            self._build_runtime_index()
            self.index_ready = True
            self._save_incidents()
            self._save_embeddings()
            LOGGER.info("Vector index rebuilt. incidents=%s", self.size)

    def list_incidents(self, service_name: str | None = None) -> list[Incident]:
        items = self.incidents
        if service_name:
            items = [incident for incident in items if incident.service_name == service_name]
        return sorted(items, key=lambda item: item.timestamp, reverse=True)

    def add_incidents(self, incidents: list[Incident]) -> tuple[int, int]:
        """Persist incidents and update the vector index."""

        with self.lock:
            existing_ids = {incident.incident_id for incident in self.incidents}
            new_incidents = [incident for incident in incidents if incident.incident_id not in existing_ids]
            duplicates = len(incidents) - len(new_incidents)
            if not new_incidents:
                return 0, duplicates

            self.incidents.extend(new_incidents)
            self._save_incidents()

            if self.index_ready and self.embeddings.shape[0] == self.size - len(new_incidents):
                new_embeddings = self.embedding_service.embed_texts([build_embedding_text(incident) for incident in new_incidents])
                self.embeddings = (
                    new_embeddings
                    if self.embeddings.size == 0
                    else np.vstack([self.embeddings, new_embeddings]).astype(np.float32)
                )
                self._build_runtime_index()
                self._save_embeddings()
            else:
                self.rebuild_index()

            LOGGER.info("Logs ingested: %s", len(new_incidents))
            return len(new_incidents), duplicates

    def search(self, query_vector: np.ndarray, top_k: int, service_name: str | None = None) -> list[SimilarIncident]:
        """Return the most similar incidents for a query vector."""

        with self.lock:
            if self.size == 0:
                return []
            if not self.index_ready:
                self.rebuild_index()
            if self.embeddings.size == 0:
                return []

            search_k = min(max(top_k * 5, top_k), self.size)
            similarities, indices = self._search_vectors(query_vector, search_k)

            matches: list[SimilarIncident] = []
            for similarity, index in zip(similarities, indices, strict=False):
                if index < 0 or index >= self.size:
                    continue
                incident = self.incidents[index]
                if service_name and incident.service_name != service_name:
                    continue
                matches.append(
                    SimilarIncident(
                        incident_id=incident.incident_id,
                        service_name=incident.service_name,
                        similarity=round(float(max(0.0, min(1.0, similarity))), 4),
                        resolution=incident.resolution_notes,
                        severity=incident.severity,
                        error_message=incident.error_message,
                    )
                )
                if len(matches) >= top_k:
                    break
            return matches

    def _search_vectors(self, query_vector: np.ndarray, top_k: int) -> tuple[list[float], list[int]]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if self.backend == "faiss" and self.index is not None:
            scores, indices = self.index.search(query, top_k)
            return scores[0].tolist(), indices[0].tolist()

        scores = np.dot(self.embeddings, query[0])
        indices = np.argsort(scores)[::-1][:top_k]
        return scores[indices].tolist(), indices.tolist()
