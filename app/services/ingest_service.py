"""Upload parsing and ingestion service."""

from __future__ import annotations

from app.models import Incident
from app.utils.preprocessing import normalize_incident, parse_json_logs, parse_txt_logs


class IngestService:
    """Validate and ingest uploaded incidents."""

    def __init__(self, vector_store) -> None:
        self.vector_store = vector_store

    def ingest(self, filename: str, content: bytes) -> tuple[int, int, int]:
        suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if suffix not in {"json", "txt"}:
            raise ValueError("Only JSON and TXT files are supported.")

        if suffix == "json":
            records, invalid_records = parse_json_logs(content)
        else:
            records, invalid_records = parse_txt_logs(content)

        incidents: list[Incident] = []
        for index, record in enumerate(records, start=1):
            try:
                incidents.append(normalize_incident(record, fallback_id=f"INGEST-{self.vector_store.size + index:05d}"))
            except ValueError:
                invalid_records += 1

        added, duplicates = self.vector_store.add_incidents(incidents)
        return added, duplicates, invalid_records
