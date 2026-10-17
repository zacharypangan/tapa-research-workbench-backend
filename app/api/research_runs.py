"""Document graphs using the hosted Workbench repository and persistent jobs."""

from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.repository.knowledge_pipeline.jobs import get_scheduler
from app.repository.knowledge_pipeline.service import ResearchPipeline, validate_config

router = APIRouter(prefix="/research-runs", tags=["research-knowledge-graphs"])


class DocumentRef(BaseModel):
    material_id: str = Field(min_length=1, max_length=255)
    file_id: str = Field(min_length=1, max_length=255)


class RunCreate(BaseModel):
    documents: list[DocumentRef] = Field(min_length=1, max_length=20)
    config: dict = Field(default_factory=dict)
    title: str | None = Field(default=None, min_length=1, max_length=200)


class AnnotationSubmission(BaseModel):
    responses: dict[str, dict]


class FrontierItem(BaseModel):
    url: str = Field(min_length=1, max_length=4096)
    seed_ids: list[str] = Field(min_length=1, max_length=100)
    question: str = Field(default="", max_length=2000)
    discovery_method: str = Field(default="registered_url", max_length=100)
    query: str = Field(default="", max_length=2000)
    discovery_locator: str = Field(default="", max_length=4096)
    selectors: str | None = Field(default=None, max_length=2000)


class FrontierSubmission(BaseModel):
    frontier: list[FrontierItem] = Field(min_length=1, max_length=25)


class SkipEnrichmentRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=2000)


class Decision(BaseModel):
    candidate_id: str = Field(min_length=1, max_length=255)
    decision: Literal["accept", "reject"]
    rationale: str = Field(min_length=1, max_length=4000)


class ReviewSubmission(BaseModel):
    decisions: list[Decision] = Field(min_length=1, max_length=1000)


class ReleaseRequest(BaseModel):
    allow_pending: bool = False


ACTOR = "workbench-researcher"


def pipeline_for() -> ResearchPipeline:
    return ResearchPipeline(get_scheduler().store)


def mapped_error(exc):
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, "Research run or artifact not found")
    return HTTPException(422, str(exc))


def resolve_documents(refs: list[DocumentRef]) -> list[dict]:
    from app.api import repository
    documents = []
    root = Path(repository.FILES_ROOT).resolve()
    with repository.get_connection() as connection:
        for ref in refs:
            row = connection.execute("SELECT f.stored_path, f.original_filename, m.title FROM files f JOIN materials m ON m.id=f.material_id WHERE f.id=? AND f.material_id=?", (ref.file_id, ref.material_id)).fetchone()
            if row is None:
                raise FileNotFoundError("Registered document not found")
            path = Path(row["stored_path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Document must be in Workbench managed file storage")
            documents.append({"path": path, "title": row["title"] or row["original_filename"],
                              "locator": f"workbench:material/{ref.material_id}/file/{ref.file_id}"})
    return documents


async def queue_run(run_id: str) -> dict:
    try:
        scheduler = get_scheduler()
        queued = await run_in_threadpool(scheduler.enqueue, run_id)
        scheduler.wake()
        return queued
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc


@router.post("", status_code=201)
async def create_run(body: RunCreate, idempotency_key: str | None = Header(default=None, max_length=255)):
    try:
        config = validate_config(body.config)
        documents = await run_in_threadpool(resolve_documents, body.documents)
        pipeline = pipeline_for()
        run = await run_in_threadpool(pipeline.store.create, documents, config, ACTOR, idempotency_key, body.title)
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc
    # Idempotency protects input snapshots; queue IDs protect each execution. Resume is explicit.
    if run["status"] != "created":
        return run
    queued = await queue_run(run["run_id"])
    return {**pipeline.store.describe(run["run_id"]), **queued}


@router.get("")
def list_runs(limit: int = Query(default=20, ge=1, le=100), offset: int = Query(default=0, ge=0, le=100000),
              material_id: str | None = Query(default=None, min_length=1, max_length=255, pattern=r"^[^/]+$")):
    return pipeline_for().store.list_runs(limit, offset, material_id=material_id)


@router.get("/capabilities")
def get_capabilities():
    from app.api import repository
    from app.repository.knowledge_pipeline.runtime import APPROVED_LOCAL_RUNTIME
    configured = repository.ai_chat_configured()
    collection_available = (find_spec("bs4") is not None and find_spec("httpx") is not None) or APPROVED_LOCAL_RUNTIME.is_file()
    return {
        "can_write": True,
        "can_review": True,
        "extraction": {
            "configured": configured,
            "default_mode": "ollama" if configured else "manual",
            "modes": [{"id": "ollama", "label": "Extract concepts automatically", "available": configured},
                      {"id": "manual", "label": "Add concepts from source text", "available": True}],
            "detail": "Model availability is checked when extraction runs; configuration does not confirm a live connection.",
        },
        "collection": {"available": collection_available, "engine": "static",
                       "label": "Collect from linked public web pages",
                       "detail": "Public web pages must be linked to concepts from your documents." if collection_available
                       else "Web collection needs the approved BeautifulSoup runtime in the worker environment."},
        "supported_formats": ["pdf", "txt", "md", "json", "docx"],
        "limits": {"max_documents": 20, "max_document_bytes": 50 * 1024 * 1024,
                   "max_input_bytes": 100 * 1024 * 1024, "max_pages": 25, "default_pages": 5},
        "statuses": {"created": "Ready to start", "queued": "Waiting to start", "running": "Processing",
                     "awaiting_annotations": "Concept extraction needs input", "awaiting_frontier": "Add web sources",
                     "awaiting_readable_documents": "Readable text needed", "awaiting_review": "Ready for review",
                     "released": "Published graph", "failed": "Needs attention", "queue_failed": "Queue unavailable"},
    }


@router.get("/{run_id}")
def get_run(run_id: str):
    try:
        return pipeline_for().store.describe(run_id)
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc


@router.get("/{run_id}/artifacts/{name:path}")
def download_artifact(run_id: str, name: str):
    try:
        path = pipeline_for().store.artifact(run_id, name)
        return FileResponse(path, filename=path.name)
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc


@router.post("/{run_id}/resume", status_code=202)
async def resume_run(run_id: str):
    try:
        run = pipeline_for().store.describe(run_id)
        if run["status"] == "released":
            raise ValueError("Released runs are immutable; create a new run")
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc
    return {**run, **await queue_run(run_id)}


@router.post("/{run_id}/annotations")
def submit_annotations(run_id: str, body: AnnotationSubmission):
    try:
        if len(body.responses) > 100 or len(body.model_dump_json()) > 5 * 1024 * 1024:
            raise ValueError("Submit at most 100 annotation batches and 5 MiB per request")
        return pipeline_for().submit_responses(run_id, body.responses)
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc


@router.post("/{run_id}/frontier")
def submit_frontier(run_id: str, body: FrontierSubmission):
    try:
        return pipeline_for().submit_frontier(run_id, [item.model_dump() for item in body.frontier], ACTOR)
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc


@router.post("/{run_id}/skip-enrichment")
def skip_enrichment(run_id: str, body: SkipEnrichmentRequest):
    try:
        return pipeline_for().skip_augmentation(run_id, body.reason, ACTOR)
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc


@router.post("/{run_id}/review")
def review_candidates(run_id: str, body: ReviewSubmission):
    try:
        return pipeline_for().review(run_id, [item.model_dump() for item in body.decisions], ACTOR)
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc


@router.post("/{run_id}/release", status_code=201)
def release_run(run_id: str, body: ReleaseRequest):
    try:
        return pipeline_for().release(run_id, ACTOR, body.allow_pending)
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc
