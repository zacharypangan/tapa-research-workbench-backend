"""English reading copies with source-bound progress and explicit review labels."""
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.repository.translations import get_translation_scheduler
from app.repository.processing_jobs import material_inputs

router = APIRouter(prefix="/repository", tags=["document-translations"])


class TranslationRequest(BaseModel):
    file_id: str | None = Field(default=None, min_length=1, max_length=255)


@router.post("/materials/{material_id}/translations", status_code=202)
async def create_translation(material_id: str, payload: TranslationRequest):
    try:
        return await run_in_threadpool(get_translation_scheduler().enqueue, material_id, payload.file_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, "Translation requires current, supported files attached to this source.") from exc


@router.get("/materials/{material_id}/translations")
def material_translations(material_id: str):
    from app.api import repository
    scheduler = get_translation_scheduler()
    with repository.get_connection() as connection:
        repository.ensure_material(connection, material_id)
    jobs = scheduler.list_for(material_id)
    current = material_inputs(material_id) if jobs else None
    for job in jobs:
        job["stale"] = scheduler.is_stale(job, current)
    return {"material_id": material_id, "jobs": jobs}


@router.get("/translation-jobs/{job_id}")
def translation_job(job_id: str):
    try:
        return get_translation_scheduler().get(job_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, "Translation job not found.") from exc


@router.post("/translation-jobs/{job_id}/retry", status_code=202)
async def retry_translation(job_id: str):
    try:
        return await run_in_threadpool(get_translation_scheduler().retry, job_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, "Translation job or current source file not found.") from exc


@router.get("/translation-jobs/{job_id}/artifact")
def translation_artifact(job_id: str, format: Literal["json", "text"] = Query(default="json")):
    try:
        scheduler = get_translation_scheduler()
        artifact = scheduler.artifact(job_id)
        artifact["stale"] = scheduler.is_stale(scheduler.get(job_id))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, "Translation artifact not found.") from exc
    if format == "text":
        lines = [artifact["label"], "Original documents remain the authoritative evidence.",
                 "Source changed: this is a historical translation." if artifact["stale"] else "", ""]
        lines.extend("Warning: " + warning for warning in artifact["warnings"])
        lines.append("")
        for unit in artifact["units"]:
            lines.extend([f"{unit['filename']} — {unit['locator']} [{unit['start']}:{unit['end']}]", "",
                          unit.get("translation_text") or f"[Translation {unit['status']}: {unit.get('error', 'not completed')}]"])
        return Response("\n".join(lines), media_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="english-reading-copy-{job_id}.txt"'})
    return artifact
