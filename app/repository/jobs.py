"""Shared durable job submission for Repository and Progress."""

import os
from datetime import datetime, timezone

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.progress.models import AuditEvent, ProgressJob

JOB_FUNCTIONS = {
    "repository_extract": "extract_repository_material",
    "repository_image_index": "index_repository_images",
    "repository_index": "index_repository_material",
}


async def enqueue_repository_job(session: Session, job_type: str, target_id: str, payload: dict, actor_id: str) -> ProgressJob:
    function = JOB_FUNCTIONS[job_type]
    job = ProgressJob(job_type=job_type, target_id=target_id, payload=payload, queued_by=actor_id)
    session.add(job)
    session.flush()
    session.add(AuditEvent(entity_type="progress_job", entity_id=job.id, action="queued", actor_id=actor_id, changes={"material_id": target_id, **payload}))
    session.commit()
    pool = None
    try:
        pool = await create_pool(RedisSettings.from_dsn(os.getenv("REDIS_PRIVATE_URL") or os.getenv("REDIS_URL", "redis://localhost:6379/0")))
        if await pool.enqueue_job(function, job.id, _job_id=job.id) is None:
            raise RuntimeError("Job was not accepted by the queue.")
    except Exception as exc:
        job.status = "queue_failed"
        job.error_message = "Job queue is unavailable. Please retry."
        job.finished_at = datetime.now(timezone.utc)
        session.commit()
        raise HTTPException(status_code=503, detail={"message": job.error_message, "job_id": job.id}) from exc
    finally:
        if pool is not None:
            await pool.close()
    return job
