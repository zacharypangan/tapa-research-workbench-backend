"""Preparation progress and retry actions for the existing document workbench."""

from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.repository.processing_jobs import STAGES, enqueue_material, get_processing_scheduler

router = APIRouter(prefix="/repository", tags=["document-preparation"])


class ProcessRequest(BaseModel):
    stages: list[Literal["extract", "graph", "text_index", "image_index"]] = Field(default_factory=lambda: list(STAGES), min_length=1, max_length=4)
    force: bool = False


@router.post("/materials/{material_id}/process", status_code=202)
async def process_material(material_id: str, payload: ProcessRequest):
    try:
        return await run_in_threadpool(enqueue_material, material_id, payload.stages, payload.force)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/materials/{material_id}/processing")
def material_processing(material_id: str):
    from app.api import repository
    with repository.get_connection() as connection:
        repository.ensure_material(connection, material_id)
        coverage = repository.embedding_coverage(connection, material_id=material_id)
    try:
        job = get_processing_scheduler().current_for(material_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, "Preparation replacement could not be resolved") from exc
    return {"material_id": material_id, "job": job, "coverage": coverage}


@router.get("/processing-jobs/{job_id}")
def get_processing_job(job_id: str):
    try:
        return get_processing_scheduler().get(job_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, "Processing job not found") from exc


@router.post("/processing-jobs/{job_id}/retry", status_code=202)
def retry_processing_job(job_id: str):
    try:
        return get_processing_scheduler().retry(job_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, "Processing job not found") from exc
