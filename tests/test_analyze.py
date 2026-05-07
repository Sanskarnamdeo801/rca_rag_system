"""Analyze endpoint and root page tests for the MVP."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_serves_ui(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "RCA MVP" in response.text


def test_analyze_returns_template_rca_with_similar_incidents(client: TestClient) -> None:
    response = client.post(
        "/analyze",
        json={"log": "Database connection timeout while processing card charge in payment-service"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["root_cause"]
    assert body["suggested_fix"]
    assert body["similar_incidents"]
    assert body["similar_incidents"][0]["incident_id"] == "INC100"
