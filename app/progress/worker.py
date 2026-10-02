"""Sequential, restart-safe extraction jobs sharing the API's local storage."""

import os
from datetime import datetime, timezone

from arq.connections import RedisSettings
from fastapi import HTTPException

from app.api.repository import build_image_index, build_semantic_index, extract_material_text, init_repository_db
from app.progress.database import SessionLocal
from app.progress.models import ProgressJob
from app.repository.jobs import JOB_FUNCTIONS
from app.repository.schemas import BuildSemanticIndexRequest, ExtractRequest
from sqlalchemy import select


async def prepare_search_index(payload: dict, images_only: bool = False) -> dict:
    """Walk every batch, keeping extracted evidence usable during provider outages."""
    result = {"warnings": [], "indexing_status": "completed"}
    builders = [("image_index", build_image_index, "last_image_id")]
    if not images_only:
        builders.insert(0, ("text_index", build_semantic_index, "last_segment_id"))
    request = BuildSemanticIndexRequest(**payload)
    for name, builder, cursor_key in builders:
        counts = {"indexed_count": 0, "captioned_count": 0}
        cursor = None
        try:
            while True:
                batch = await builder(request, after_id=cursor)
                if not batch.get("provider_configured", True):
                    result["indexing_status"] = "deferred"
                    result["warnings"].append("AI indexing is deferred. Exact search remains available; retry indexing when the embedding service is ready.")
                    break
                for key in counts:
                    counts[key] += batch.get(key, 0)
                if batch.get("embedding_configured") is False:
                    result["indexing_status"] = "deferred"
                if batch.get("caption_failed_count"):
                    result["indexing_status"] = "deferred"
                    result["warnings"].append("Some image descriptions are unavailable. OCR and source context remain searchable; retry indexing when the vision model is ready.")
                next_cursor = batch.get(cursor_key)
                if next_cursor is None or next_cursor == cursor:
                    break
                cursor = next_cursor
        except Exception:
            result["indexing_status"] = "deferred"
            result["warnings"].append(f"{name.replace('_', ' ').capitalize()} is incomplete. Extracted evidence was preserved; retry indexing when the AI service is ready.")
        result[name] = counts
    result["warnings"] = list(dict.fromkeys(result["warnings"]))
    return result


async def run_repository_job(job_id: str, images_only: bool = False, index_only: bool = False) -> dict:
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
        if images_only or index_only:
            result = await prepare_search_index(payload, images_only=images_only)
        else:
            result = await extract_material_text(
                material_id=material_id,
                payload=ExtractRequest(**payload.get("extract", {"include_links": False})),
                force=bool(payload.get("force", False)),
                run_id=job_id,
            )
            index = await prepare_search_index({"material_id": material_id, "limit": 100})
            result.setdefault("warnings", []).extend(index.pop("warnings"))
            result.update(index)
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


async def index_repository_material(context: dict, job_id: str) -> dict:
    return await run_repository_job(job_id, index_only=True)


async def startup(context: dict):
    init_repository_db()
    with SessionLocal() as session:
        jobs = list(session.scalars(select(ProgressJob).where(ProgressJob.status.in_(["queued", "running"])).order_by(ProgressJob.created_at)))
    for job in jobs:
        function = JOB_FUNCTIONS.get(job.job_type)
        if function:
            await context["redis"].enqueue_job(function, job.id, _job_id=job.id)


class WorkerSettings:
    functions = [extract_repository_material, index_repository_images, index_repository_material]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(os.getenv("REDIS_PRIVATE_URL") or os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    max_jobs = 1
    log_results = False
    health_check_interval = 15
    job_timeout = int(os.getenv("PROGRESS_JOB_TIMEOUT_SECONDS", "3600"))
    max_tries = 3
