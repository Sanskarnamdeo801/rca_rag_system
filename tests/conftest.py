"""Shared fixtures for the simplified RCA MVP tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture()
def sample_incidents() -> list[dict[str, str]]:
    """Return deterministic sample incidents for tests."""

    return [
        {
            "incident_id": "INC100",
            "timestamp": "2026-01-10T12:30:22",
            "service_name": "payment-service",
            "log_level": "ERROR",
            "error_message": (
                "Database connection timeout while processing card charge"
            ),
            "stack_trace": (
                "DBConnectionTimeoutException at PaymentRepository.save"
            ),
            "resolution_notes": (
                "Restart PostgreSQL and increase the timeout"
            ),
            "severity": "HIGH",
        },
        {
            "incident_id": "INC101",
            "timestamp": "2026-01-10T13:30:22",
            "service_name": "order-service",
            "log_level": "ERROR",
            "error_message": (
                "Redis cache connection reset by peer during order lookup"
            ),
            "stack_trace": "RedisConnectionError at OrderCache.load",
            "resolution_notes": (
                "Restart Redis and flush the client pool"
            ),
            "severity": "HIGH",
        },
    ]


@pytest.fixture()
def settings(
    tmp_path: Path,
    sample_incidents: list[dict[str, str]],
) -> Settings:
    """Create isolated test settings."""

    data_dir = tmp_path / "data"
    sample_path = tmp_path / "seed_incidents.json"

    sample_path.write_text(
        json.dumps(sample_incidents),
        encoding="utf-8",
    )

    settings = Settings(
        debug=True,
        data_dir=data_dir,
        incidents_path=data_dir / "incidents.json",
        faiss_dir=data_dir / "faiss_index",
        faiss_index_path=data_dir / "faiss_index" / "index.faiss",
        embeddings_path=data_dir / "faiss_index" / "embeddings.npy",
        static_dir=(
            Path(__file__).resolve().parents[1]
            / "app"
            / "static"
        ),
        sample_incidents_path=sample_path,
        embedding_backend="hashing",
        embedding_warmup=False,
        llm_provider="template",
    )

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.faiss_dir.mkdir(parents=True, exist_ok=True)

    return settings


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    """Return FastAPI test client."""

    with TestClient(create_app(settings)) as test_client:
        yield test_client