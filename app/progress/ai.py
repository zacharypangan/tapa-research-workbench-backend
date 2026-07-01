"""Evidence-constrained candidate extraction for the progress ledger."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import httpx

from app.api.repository import (
    OLLAMA_RETRIEVAL_MODEL,
    ai_chat_configured,
    ollama_endpoint,
    ollama_headers,
    ollama_provider_name,
)


PROMPT_VERSION = "progress_candidate_v1"
RECORD_TYPES = {
    "member_progress",
    "dataset",
    "spatial_issue",
    "temporal_issue",
    "uncertainty_issue",
    "method_claim",
    "technique",
    "decision",
    "open_question",
    "action_item",
    "repository_enrichment",
}


def build_candidate_prompt(sources: list[dict[str, str]]) -> str:
    evidence = "\n\n".join(
        f"SOURCE_ID: {source['id']}\nLANGUAGE: {source['language']}\nTEXT: {source['text']}"
        for source in sources
    )
    return f"""You extract reviewable research-progress records from meeting evidence.
Return JSON only as an array. Each object must contain:
- record_type: one of {sorted(RECORD_TYPES)}
- title_original, title_en, title_ja (nullable strings)
- summary_en, summary_ja (nullable strings)
- confidence (0 to 1)
- evidence_source_ids (one or more SOURCE_ID values)
- details (object)

Rules:
1. Do not invent facts or merge unrelated speakers.
2. Preserve original-language wording in title_original where possible.
3. Decisions, action items, and accepted framework rules require explicit evidence; otherwise use open_question or method_claim.
4. Every output is only a candidate for human review.
5. Focus on member progress, datasets, space, time, uncertainty, provenance, GIS, OCR, AI extraction, ontology, knowledge graphs, and governance.

EVIDENCE:
{evidence}
"""


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def parse_candidate_response(raw: str) -> list[dict[str, Any]]:
    value = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", value, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1).strip()
    payload = json.loads(value)
    if not isinstance(payload, list):
        raise ValueError("AI extraction response must be a JSON array.")
    candidates: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict) or item.get("record_type") not in RECORD_TYPES:
            continue
        source_ids = item.get("evidence_source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            continue
        item["confidence"] = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
        item["details"] = item.get("details") if isinstance(item.get("details"), dict) else {}
        candidates.append(item)
    return candidates


async def request_candidate_extraction(sources: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    if not ai_chat_configured():
        raise RuntimeError("Repository AI generation is not configured.")
    prompt = build_candidate_prompt(sources)
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            ollama_endpoint("chat"),
            headers=ollama_headers(),
            json={
                "model": OLLAMA_RETRIEVAL_MODEL,
                "messages": [
                    {"role": "system", "content": "Return valid JSON only. Never accept records automatically."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.1},
            },
        )
    response.raise_for_status()
    raw = response.json().get("message", {}).get("content", "")
    return raw, parse_candidate_response(raw)


def provider_metadata() -> tuple[str, str]:
    return ollama_provider_name(), OLLAMA_RETRIEVAL_MODEL
