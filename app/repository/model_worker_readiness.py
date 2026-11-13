"""Read production text-worker status without submitting inference."""
from __future__ import annotations

import math
import time
from typing import NamedTuple

import httpx

from app.repository.model_tasks import READINESS_VERSION, SUCCESS_MAX_AGE_SECONDS, WORKER_MAX_AGE_SECONDS


PRODUCTION_GATEWAY = "https://tapa-research-workbench-backend-production.up.railway.app/api/v1/model-gateway"
MINPAKU_TEXT_MODEL = "qwen3:8b"


class WorkerReadiness(NamedTuple):
    connected: bool
    verified: bool
    detail: str


def _number(value):
    return type(value) in (int, float) and abs(value) <= 10**15 and math.isfinite(value)


async def minpaku_text_readiness(headers: dict) -> WorkerReadiness:
    """Check that the model can accept work, independently of recent execution."""
    unavailable = WorkerReadiness(False, False, "Minpaku text worker status could not be verified.")
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            response = await client.get(PRODUCTION_GATEWAY + "/health", headers=headers)
        if response.status_code != 200 or len(response.content) > 64 * 1024:
            return unavailable
        health = response.json()
        if not isinstance(health, dict) or health.get("version") != READINESS_VERSION:
            return unavailable
        checked = health.get("checked_at")
        worker_age, success_age = health.get("worker_max_age_seconds"), health.get("success_max_age_seconds")
        queued, running = health.get("queued_count"), health.get("running_count")
        if (not _number(checked) or not -5 <= time.time() - checked <= 10
                or not _number(worker_age) or not 0 < worker_age <= WORKER_MAX_AGE_SECONDS
                or not _number(success_age) or not 0 < success_age <= SUCCESS_MAX_AGE_SECONDS
                or type(queued) is not int or not 0 <= queued <= 32
                or type(running) is not int or not 0 <= running <= 32):
            return unavailable
        models = health.get("models")
        row = models.get(MINPAKU_TEXT_MODEL) if isinstance(models, dict) else None
        if not isinstance(row, dict):
            return unavailable
        queue = f"Queue: {queued} waiting, {running} running."
        seen, succeeded = row.get("last_seen_at"), row.get("last_success_at")
        if row.get("worker_fresh") is not True or not _number(seen) or not 0 <= checked - seen <= worker_age:
            return WorkerReadiness(False, False, "Minpaku text worker is disconnected or its heartbeat is stale. " + queue)
        if (row.get("ready") is not True or not _number(succeeded) or not 0 <= checked - succeeded <= success_age
                or row.get("last_success_operation") not in ("chat", "generate")):
            return WorkerReadiness(True, False, "Minpaku text worker is connected; recent inference is unverified. " + queue)
        return WorkerReadiness(True, True, "Minpaku text worker is connected and recently completed inference. " + queue)
    except (httpx.HTTPError, ValueError, TypeError, KeyError, OverflowError):
        return unavailable
