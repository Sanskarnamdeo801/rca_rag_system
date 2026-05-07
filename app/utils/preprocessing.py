"""Minimal preprocessing helpers for ingestion and retrieval."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from app.models import Incident

_ID_PATTERN = re.compile(r"\b(?:request_id|trace_id|correlation_id|span_id)=[\w-]+\b", re.IGNORECASE)
_IP_PATTERN = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_SPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize operational log text for embeddings and matching."""

    cleaned = _ID_PATTERN.sub(" ", text)
    cleaned = _IP_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.replace("\\n", " ").replace("\n", " ")
    cleaned = _SPACE_PATTERN.sub(" ", cleaned).strip().lower()
    return cleaned


def infer_severity(text: str) -> str:
    """Infer severity from the message when no historical signal exists."""

    normalized = normalize_text(text)
    if any(token in normalized for token in ("critical", "outage", "memory leak", "eviction")):
        return "CRITICAL"
    if any(token in normalized for token in ("timeout", "failed", "unavailable", "error", "exhaustion")):
        return "HIGH"
    if any(token in normalized for token in ("warning", "latency", "retry")):
        return "MEDIUM"
    return "LOW"


def build_embedding_text(incident: Incident) -> str:
    """Create a compact semantic representation for search."""

    return normalize_text(
        f"{incident.service_name} {incident.log_level} {incident.error_message} {incident.stack_trace} {incident.resolution_notes}"
    )


def normalize_incident(record: dict[str, Any], fallback_id: str) -> Incident:
    """Normalize incoming incident payloads into the canonical model."""

    timestamp = record.get("timestamp") or datetime.now(timezone.utc).isoformat()
    incident_id = str(record.get("incident_id") or fallback_id)
    service_name = str(record.get("service_name") or "unknown-service").strip()
    error_message = str(record.get("error_message") or record.get("message") or "").strip()
    if not error_message:
        raise ValueError("error_message is required")

    return Incident(
        incident_id=incident_id,
        timestamp=timestamp,
        service_name=service_name,
        log_level=str(record.get("log_level") or "ERROR"),
        error_message=error_message,
        stack_trace=str(record.get("stack_trace") or ""),
        resolution_notes=str(record.get("resolution_notes") or ""),
        severity=str(record.get("severity") or infer_severity(error_message)),
        tags=list(record.get("tags") or []),
    )


def parse_json_logs(raw_bytes: bytes) -> tuple[list[dict[str, Any]], int]:
    """Parse uploaded JSON bytes into a list of incident payloads."""

    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid JSON file.") from exc

    if isinstance(payload, dict):
        return [payload], 0
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)], len([item for item in payload if not isinstance(item, dict)])
    raise ValueError("JSON payload must be an object or list of objects.")


def parse_txt_logs(raw_bytes: bytes) -> tuple[list[dict[str, Any]], int]:
    """Parse key-value text log bundles separated by blank lines."""

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Invalid text file encoding.") from exc

    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    records: list[dict[str, Any]] = []
    invalid = 0

    for block in blocks:
        record: dict[str, Any] = {}
        current_key: str | None = None
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                current_key = key.strip()
                record[current_key] = value.strip()
            elif current_key:
                record[current_key] = f"{record[current_key]}\n{line.strip()}".strip()
        if record:
            records.append(record)
        else:
            invalid += 1

    return records, invalid
