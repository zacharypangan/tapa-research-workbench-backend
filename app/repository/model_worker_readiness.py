"""Metadata-only readiness for the exact approved staging model bridge.

This helper never submits inference and never redirects authorization to another
origin. Other configured model providers retain their existing health probes.
"""
from __future__ import annotations

import math
import time

import httpx

from app.repository.model_tasks import READINESS_VERSION, SUCCESS_MAX_AGE_SECONDS, WORKER_MAX_AGE_SECONDS


STAGING_GATEWAY = "https://tapa-research-workbench-backend-graph-staging.up.railway.app/api/v1/model-gateway"
_OPERATIONS = {"qwen3:8b": {"chat", "generate"},
               "qwen3-vl:8b-instruct": {"chat", "generate"},
               "qwen3-embedding:0.6b": {"embed", "embeddings"}}


def _number(value):
    return type(value) in (int, float) and abs(value) <= 10**15 and math.isfinite(value)


async def bridge_readiness(base_url: str, headers: dict, model: str) -> tuple[bool, str] | None:
    """Require a fresh worker and matching recent execution on that same worker.

    Busy queues do not cause synthetic health jobs to expire and wrongly disable
    the platform. Readiness proves recent execution, not scientific answer quality
    or that the next request will finish before its deadline.
    """
    if not isinstance(base_url, str) or base_url.rstrip("/") not in {STAGING_GATEWAY, STAGING_GATEWAY + "/api"}:
        return None
    if not isinstance(model, str) or model not in _OPERATIONS:
        return False, "This model is not approved for the private staging bridge."
    unavailable = "Private model worker readiness could not be verified; no inference was submitted."
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            response = await client.get(STAGING_GATEWAY + "/health", headers=headers)
        if response.status_code != 200 or len(response.content) > 64 * 1024:
            return False, unavailable
        health = response.json()
        if not isinstance(health, dict) or health.get("version") != READINESS_VERSION:
            return False, unavailable
        checked = health.get("checked_at")
        worker_age, success_age = health.get("worker_max_age_seconds"), health.get("success_max_age_seconds")
        queued, running = health.get("queued_count"), health.get("running_count")
        if (not _number(checked) or not -5 <= time.time() - checked <= 10
                or not _number(worker_age) or not 0 < worker_age <= WORKER_MAX_AGE_SECONDS
                or not _number(success_age) or not 0 < success_age <= SUCCESS_MAX_AGE_SECONDS
                or type(queued) is not int or not 0 <= queued <= 32
                or type(running) is not int or not 0 <= running <= 32):
            return False, unavailable
        models = health.get("models")
        row = models.get(model) if isinstance(models, dict) else None
        if not isinstance(row, dict):
            return False, unavailable
        queue = f"Queue: {queued} waiting, {running} running."
        seen, succeeded = row.get("last_seen_at"), row.get("last_success_at")
        worker_fresh = row.get("worker_fresh") is True and _number(seen) and 0 <= checked - seen <= worker_age
        if not worker_fresh:
            return False, "Private model worker is disconnected or its heartbeat is stale. " + queue
        execution_fresh = (_number(succeeded) and 0 <= checked - succeeded <= success_age
                           and row.get("last_success_operation") in _OPERATIONS[model])
        if row.get("ready") is not True or not execution_fresh:
            return False, "Private model worker is connected, but recent successful execution for this model is unverified. " + queue
        age = int(max(0, checked - succeeded))
        return True, (f"Private model worker heartbeat is fresh; successful {row['last_success_operation']} "
                      f"execution was verified {age} seconds ago. " + queue)
    except (httpx.HTTPError, ValueError, TypeError, KeyError, OverflowError):
        return False, unavailable
