"""ARQ worker for restart-safe progress and repository jobs."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from arq.connections import RedisSettings
from fastapi import BackgroundTasks

from app.api.repository import extract_material_text
from app.progress.database import SessionLocal
from app.progress.models import ProgressJob
from app.repository.schemas import ExtractRequest


async def extract_repository_material(context: dict, job_id: str) -> dict:
    del context
    with SessionLocal() as session:
        job = session.get(ProgressJob, job_id)
        if not job:
            raise ValueError(f"Progress job not found: {job_id}")
        job.status = "running"
        job.attempts += 1
        job.started_at = datetime.now(timezone.utc)
        session.commit()
        material_id = job.target_id
        force = bool(job.payload.get("force", False))

    try:
        result = await extract_material_text(
            material_id=material_id,
            background_tasks=BackgroundTasks(),
            payload=ExtractRequest(include_links=False),
            force=force,
        )
    except Exception as exc:
        with SessionLocal() as session:
            job = session.get(ProgressJob, job_id)
            if job:
                job.status = "failed"
                job.error_message = str(exc)
                job.finished_at = datetime.now(timezone.utc)
                session.commit()
        raise

    with SessionLocal() as session:
        job = session.get(ProgressJob, job_id)
        if job:
            job.status = "completed"
            job.result = result if isinstance(result, dict) else {"result": str(result)}
            job.finished_at = datetime.now(timezone.utc)
            session.commit()
    return result


class WorkerSettings:
    functions = [extract_repository_material]
    redis_settings = RedisSettings.from_dsn(
        os.getenv("REDIS_PRIVATE_URL") or os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    max_jobs = int(os.getenv("PROGRESS_WORKER_MAX_JOBS", "2"))
    job_timeout = int(os.getenv("PROGRESS_JOB_TIMEOUT_SECONDS", "3600"))
    max_tries = 3
