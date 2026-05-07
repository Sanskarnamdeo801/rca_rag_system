"""Ingestion API tests for the MVP."""

from __future__ import annotations

import json
from io import BytesIO

from fastapi.testclient import TestClient


def test_ingest_json_file_updates_incident_store(client: TestClient) -> None:
    payload = [
        {
            "incident_id": "INC200",
            "timestamp": "2026-01-12T10:00:00",
            "service_name": "auth-service",
            "log_level": "ERROR",
            "error_message": "JWT token validation failed",
            "resolution_notes": "Rotate the signing key",
            "severity": "HIGH",
        }
    ]

    response = client.post(
        "/ingest",
        files={"file": ("logs.json", BytesIO(json.dumps(payload).encode("utf-8")), "application/json")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["logs_ingested"] == 1
    assert body["total_incidents"] >= 3
