"""Sequential, restart-safe extraction jobs sharing the API's local storage."""

import os
from datetime import datetime, timezone

from arq.connections import RedisSettings
from fastapi import HTTPException

from app.api.repository import build_image_index, extract_material_text, init_repository_db
from app.progress.database import SessionLocal
from app.progress.models import ProgressJob
from app.repository.jobs import JOB_FUNCTIONS
from app.repository.schemas import BuildSemanticIndexRequest, ExtractRequest
from sqlalchemy import select


async def run_repository_job(job_id: str, images_only: bool = False) -> dict:
    with SessionLocal() as session:
        job = session.get(ProgressJob, job_id)
        if not job:
            raise ValueError("Repository job not found")
        if job.status == "completed":
            return job.result
        job.status = "running"
        job.attempts += 1
        job.started_at = datetime.now(timezone.utc)
        job.error_message = None
        job.finished_at = None
        session.commit()
        material_id, payload = job.target_id, dict(job.payload)

    try:
        if images_only:
            result = await build_image_index(BuildSemanticIndexRequest(**payload))
        else:
            result = await extract_material_text(
                material_id=material_id,
                payload=ExtractRequest(**payload.get("extract", {"include_links": False})),
                force=bool(payload.get("force", False)),
                run_id=job_id,
            )
            if result.get("image_evidence_count"):
                try:
                    result["image_index"] = await build_image_index(BuildSemanticIndexRequest(material_id=material_id, limit=1000))
                except Exception:
                    result["warnings"].append("Image descriptions were unavailable. Extracted source evidence was preserved.")
    except Exception as exc:
        with SessionLocal() as session:
            job = session.get(ProgressJob, job_id)
            job.status = "failed"
            job.error_message = str(exc.detail) if isinstance(exc, HTTPException) else "Extraction failed. Please retry or upload a new source version."
            job.finished_at = datetime.now(timezone.utc)
            session.commit()
        raise

    with SessionLocal() as session:
        job = session.get(ProgressJob, job_id)
        job.status = "completed"
        job.result = result
        job.finished_at = datetime.now(timezone.utc)
        session.commit()
    return result


async def extract_repository_material(context: dict, job_id: str) -> dict:
    return await run_repository_job(job_id)


async def index_repository_images(context: dict, job_id: str) -> dict:
    return await run_repository_job(job_id, images_only=True)


async def startup(context: dict):
    init_repository_db()
    with SessionLocal() as session:
        jobs = list(session.scalars(select(ProgressJob).where(ProgressJob.status.in_(["queued", "running"])).order_by(ProgressJob.created_at)))
    for job in jobs:
        function = JOB_FUNCTIONS.get(job.job_type)
        if function:
            await context["redis"].enqueue_job(function, job.id, _job_id=job.id)


class WorkerSettings:
    functions = [extract_repository_material, index_repository_images]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(os.getenv("REDIS_PRIVATE_URL") or os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    max_jobs = 1
    log_results = False
    health_check_interval = 15
    job_timeout = int(os.getenv("PROGRESS_JOB_TIMEOUT_SECONDS", "3600"))
    max_tries = 3
