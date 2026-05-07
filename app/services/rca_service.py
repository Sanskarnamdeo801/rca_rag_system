"""RCA generation with graceful LLM fallback."""

from __future__ import annotations

import json

import httpx

from app.config import Settings
from app.models import RCAResponse, SimilarIncident
from app.utils.logger import get_logger
from app.utils.preprocessing import infer_severity

LOGGER = get_logger(__name__)


class RCAService:
    """Run retrieval and generate RCA responses."""

    def __init__(self, settings: Settings, retrieval_service) -> None:
        self.settings = settings
        self.retrieval_service = retrieval_service

    async def analyze(self, log: str, service_name: str | None = None, top_k: int | None = None) -> RCAResponse:
        matches = self.retrieval_service.find_similar(
            log,
            service_name=service_name,
            top_k=top_k or self.settings.top_k_results,
        )
        fallback = self._template_response(log, matches)
        if self.settings.llm_provider in {"template", "mock"}:
            return fallback

        try:
            llm_response = await self._generate_with_llm(log, matches)
            if not llm_response.root_cause or not llm_response.suggested_fix:
                return fallback
            return llm_response
        except Exception as exc:  # pragma: no cover - depends on provider availability
            LOGGER.error("RCA generation failed. Falling back to template. error=%s", exc)
            return fallback

    async def _generate_with_llm(self, log: str, matches: list[SimilarIncident]) -> RCAResponse:
        if self.settings.llm_provider == "openai":
            return await self._call_openai(log, matches)
        if self.settings.llm_provider == "llama3":
            return await self._call_llama3(log, matches)
        return self._template_response(log, matches)

    async def _call_openai(self, log: str, matches: list[SimilarIncident]) -> RCAResponse:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        payload = {
            "model": self.settings.openai_model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(log, matches)},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
            response = await client.post(self.settings.openai_base_url, json=payload, headers=headers)
            response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return self._parse_llm_response(content, matches)

    async def _call_llama3(self, log: str, matches: list[SimilarIncident]) -> RCAResponse:
        payload = {
            "model": self.settings.llama3_model,
            "prompt": f"{self._system_prompt()}\n\n{self._user_prompt(log, matches)}",
            "stream": False,
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
            response = await client.post(self.settings.llama3_base_url, json=payload)
            response.raise_for_status()
        content = response.json()["response"]
        return self._parse_llm_response(content, matches)

    def _parse_llm_response(self, content: str, matches: list[SimilarIncident]) -> RCAResponse:
        payload = json.loads(content)
        return RCAResponse(
            root_cause=str(payload.get("root_cause") or ""),
            severity=str(payload.get("severity") or (matches[0].severity if matches else "MEDIUM")).upper(),
            suggested_fix=str(payload.get("suggested_fix") or ""),
            confidence_score=float(payload.get("confidence_score") or self._confidence(matches)),
            similar_incidents=matches,
        )

    def _template_response(self, log: str, matches: list[SimilarIncident]) -> RCAResponse:
        if matches:
            best = matches[0]
            root_cause = (
                f"Likely caused by {best.error_message.lower()} similar to historical incident {best.incident_id}."
            )
            suggested_fix = best.resolution or "Review the affected dependency, restart the service if safe, and validate capacity."
            severity = best.severity
        else:
            root_cause = f"Likely caused by {log.strip().lower()} based on the current log pattern."
            suggested_fix = "Inspect the failing service, verify upstream dependencies, and retry after stabilizing the component."
            severity = infer_severity(log)

        return RCAResponse(
            root_cause=root_cause,
            severity=severity,
            suggested_fix=suggested_fix,
            confidence_score=self._confidence(matches),
            similar_incidents=matches,
        )

    def _confidence(self, matches: list[SimilarIncident]) -> float:
        if not matches:
            return 0.42
        return round(max(0.45, min(0.98, matches[0].similarity)), 2)

    def _system_prompt(self) -> str:
        return (
            "You are an RCA assistant. Return valid JSON with keys "
            "root_cause, severity, suggested_fix, confidence_score. "
            "Be concise and use the retrieved incidents as evidence."
        )

    def _user_prompt(self, log: str, matches: list[SimilarIncident]) -> str:
        incidents = [
            {
                "incident_id": match.incident_id,
                "service_name": match.service_name,
                "severity": match.severity,
                "similarity": match.similarity,
                "error_message": match.error_message,
                "resolution": match.resolution,
            }
            for match in matches
        ]
        return json.dumps({"log": log, "similar_incidents": incidents}, indent=2)
