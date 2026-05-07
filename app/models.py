"""Shared request and response models for the RCA MVP."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Incident(BaseModel):
    """Canonical incident persisted by the application."""

    incident_id: str = Field(min_length=3, max_length=64)
    timestamp: datetime
    service_name: str = Field(min_length=2, max_length=100)
    log_level: str = Field(default="ERROR", min_length=2, max_length=20)
    error_message: str = Field(min_length=3, max_length=4000)
    stack_trace: str = ""
    resolution_notes: str = ""
    severity: str = Field(default="MEDIUM", min_length=3, max_length=20)
    tags: list[str] = Field(default_factory=list)

    @field_validator("service_name", "incident_id", "severity", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> str:
        return str(value).strip()

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: Any) -> str:
        return str(value).strip().upper()

    @field_validator("severity", mode="after")
    @classmethod
    def normalize_severity(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            return "MEDIUM"
        return normalized


class AnalyzeRequest(BaseModel):
    """Analyze request payload."""

    log: str = Field(min_length=3, max_length=6000)
    service_name: str | None = Field(default=None, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SimilarIncident(BaseModel):
    """Returned similarity match."""

    incident_id: str
    service_name: str
    similarity: float = Field(ge=0.0, le=1.0)
    resolution: str
    severity: str
    error_message: str


class RCAResponse(BaseModel):
    """Analyze response format for the frontend."""

    root_cause: str
    severity: str
    suggested_fix: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    similar_incidents: list[SimilarIncident]


class IngestResponse(BaseModel):
    """Upload result."""

    status: str
    logs_ingested: int
    duplicates_skipped: int
    invalid_records: int
    total_incidents: int


class HealthResponse(BaseModel):
    """Health endpoint payload."""

    status: str
    incidents_count: int
    index_ready: bool
    embedding_backend: str
    llm_provider: str


class ErrorResponse(BaseModel):
    """API error response."""

    detail: str
