"""Similarity retrieval tests for the MVP."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_similar_returns_matching_incident(client: TestClient) -> None:
    response = client.get("/similar", params={"message": "database timeout in payment service", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body
    assert body[0]["incident_id"] == "INC100"
    assert body[0]["similarity"] > 0
