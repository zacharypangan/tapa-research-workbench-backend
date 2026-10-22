"""Document graphs using the hosted Workbench repository and persistent jobs."""

from importlib.util import find_spec
from pathlib import Path
import re
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.repository.document_graph_runtime import get_scheduler
from app.repository.knowledge_pipeline.service import ResearchPipeline
from app.repository.graph_generation_profiles import prepare_generation_config, generation_ontology, GENERATION_PROFILE_VERSION

router = APIRouter(prefix="/research-runs", tags=["research-knowledge-graphs"])


class DocumentRef(BaseModel):
    material_id: str = Field(min_length=1, max_length=255)
    file_id: str = Field(min_length=1, max_length=255)


class RunCreate(BaseModel):
    documents: list[DocumentRef] = Field(min_length=1, max_length=20)
    config: dict = Field(default_factory=dict)
    title: str | None = Field(default=None, min_length=1, max_length=200)


class RetryRunRequest(BaseModel):
    config: dict | None = None
    title: str | None = Field(default=None, min_length=1, max_length=200)


class ImproveRunRequest(BaseModel):
    max_auto_batches: int = Field(default=4, ge=1, le=20)


class ExtractionRunCreate(BaseModel):
    material_id: str = Field(min_length=1, max_length=255)
    file_ids: list[str] | None = Field(default=None, min_length=1, max_length=20)
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
            row = connection.execute("SELECT f.stored_path, f.original_filename, m.title, m.language FROM files f JOIN materials m ON m.id=f.material_id WHERE f.id=? AND f.material_id=?", (ref.file_id, ref.material_id)).fetchone()
            if row is None:
                raise FileNotFoundError("Registered document not found")
            path = Path(row["stored_path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Document must be in Workbench managed file storage")
            count = connection.execute("SELECT COUNT(*) FROM files WHERE material_id=?", (ref.material_id,)).fetchone()[0]
            documents.append({"path": path, "title": row["original_filename"] if count > 1 else row["title"] or row["original_filename"],
                              "locator": f"workbench:material/{ref.material_id}/file/{ref.file_id}",
                              "filename": row["original_filename"], "ocr_languages": row["language"]})
    return documents


def retry_source_metadata(store, run_id: str) -> dict:
    """Restore registered language/filename for pre-representation legacy runs."""
    from app.api import repository
    from app.repository.knowledge_pipeline.store import read_json
    descriptors = read_json(store.path(run_id) / "inputs.json")["documents"]
    result = {}
    with repository.get_connection() as connection:
        for descriptor in descriptors:
            locator = descriptor["original_locator"]
            match = re.fullmatch(r"workbench:material/([^/]+)/file/([^/]+)", locator)
            if not match:
                continue
            row = connection.execute("SELECT f.original_filename, m.language FROM files f JOIN materials m ON m.id=f.material_id WHERE m.id=? AND f.id=?", match.groups()).fetchone()
            if row:
                result[locator] = {"filename": row["original_filename"], "ocr_languages": row["language"]}
    return result


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
        config = prepare_generation_config(body.config)
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


@router.post("/from-extraction", status_code=201)
async def create_extraction_run(body: ExtractionRunCreate, idempotency_key: str | None = Header(default=None, max_length=255)):
    """Use registered text for any extracted format; preserve its original file attribution."""
    from app.repository.knowledge_pipeline.registered_extraction import registered_documents
    try:
        config = prepare_generation_config(body.config)
        documents = await run_in_threadpool(registered_documents, body.material_id, body.file_ids)
        pipeline = pipeline_for()
        run = await run_in_threadpool(pipeline.store.create, documents, config, ACTOR, idempotency_key, body.title)
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc
    if run["status"] != "created":
        return run
    return {**pipeline.store.describe(run["run_id"]), **await queue_run(run["run_id"])}


@router.get("/capabilities")
def get_capabilities():
    from app.api import repository
    from app.repository.knowledge_pipeline.runtime import APPROVED_LOCAL_RUNTIME
    configured = repository.ai_chat_configured()
    from app.repository.knowledge_pipeline.processing import provider_capabilities, PROMPT_VERSION
    from app.repository.knowledge_pipeline.ontologies import ONTOLOGY_TEMPLATES
    from app.repository.document_graph_runtime import candidate_identity, execution_identity, pipeline_identity
    compatibility = provider_capabilities(repository.effective_ollama_base_url())
    collection_available = (find_spec("bs4") is not None and find_spec("httpx") is not None) or APPROVED_LOCAL_RUNTIME.is_file()
    return {
        "can_write": True,
        "can_review": True,
        "extraction": {
            "configured": configured,
            "candidate_provider": candidate_identity(),
            "execution_identity": execution_identity(),
            "pipeline_implementation_sha256": pipeline_identity(),
            "processing_version": PROMPT_VERSION,
            "default_mode": "ollama" if configured else "manual",
            "modes": [{"id": "ollama", "label": "Extract concepts automatically", "available": configured},
                      {"id": "manual", "label": "Add concepts from source text", "available": True}],
            **compatibility,
            "detail": compatibility["detail"] + " Configuration does not confirm live extraction quality.",
        },
        "collection": {"available": collection_available, "engine": "static",
                       "discovery_methods": ["manual", "crossref"], "iterative_candidate_yield": True,
                       "label": "Collect from linked public web pages",
                       "detail": "Public web pages must be linked to concepts from your documents." if collection_available
                       else "Web collection needs the approved BeautifulSoup runtime in the worker environment."},
        "image_description": repository.image_description_diagnostics(),
        "supported_formats": ["pdf", "txt", "md", "json", "docx"],
        "registered_extraction": {"available": True, "endpoint": "/research-runs/from-extraction",
                                  "preserves_original_files": True, "supports_any_registered_extracted_format": True},
        "automatic_batch_budget": {"available": True, "per_pass": True, "maximum": 100, "default": None},
        "retry": {"creates_derived_run": True, "preserves_original": True},
        "relationship_generation": {"version": GENERATION_PROFILE_VERSION, "preserves_original": True,
                                    "endpoint": "/research-runs/{run_id}/improve", "default_batch_budget": 4},
        "ontology_templates": ONTOLOGY_TEMPLATES,
        "evidence_selections": ["span_id", "character_offsets", "exact_quote"],
        "limits": {"max_documents": 20, "max_document_bytes": 50 * 1024 * 1024,
                   "max_input_bytes": 100 * 1024 * 1024, "max_pages": 25, "default_pages": 5},
        "statuses": {"created": "Ready to start", "queued": "Waiting to start", "running": "Processing",
                     "awaiting_annotations": "Concept extraction needs input", "awaiting_frontier": "Add web sources",
                     "awaiting_readable_documents": "Readable text needed", "awaiting_review": "Ready for review",
                     "released": "Published graph", "failed": "Needs attention", "queue_failed": "Queue unavailable"},
    }


@router.get("/extraction-inventory/{material_id}")
async def get_extraction_inventory(material_id: str):
    from app.repository.document_graph_inventory import extraction_inventory
    try:
        return await run_in_threadpool(extraction_inventory, material_id)
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc


@router.get("/ontologies/{template_id}")
def get_ontology(template_id: Literal["general", "tapa"]):
    from app.repository.knowledge_pipeline.ontologies import ontology_template
    return generation_ontology(ontology_template(template_id))


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


@router.post("/{run_id}/retry", status_code=201)
async def retry_run(run_id: str, body: RetryRunRequest,
                    idempotency_key: str | None = Header(default=None, max_length=255)):
    try:
        pipeline = pipeline_for()
        metadata = await run_in_threadpool(retry_source_metadata, pipeline.store, run_id)
        from app.repository.knowledge_pipeline.store import read_json
        selected = body.config if body.config is not None else read_json(pipeline.store.path(run_id) / "inputs.json")["config"]
        config = prepare_generation_config(selected)
        run = await run_in_threadpool(pipeline.store.derive, run_id, ACTOR, config, body.title, idempotency_key, metadata)
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc
    if run["status"] != "created":
        return run
    return {**pipeline.store.describe(run["run_id"]), **await queue_run(run["run_id"])}


@router.post("/{run_id}/improve", status_code=201)
async def improve_run(run_id: str, body: ImproveRunRequest,
                      idempotency_key: str | None = Header(default=None, max_length=255)):
    """Generate a fresh, bounded source-relationship draft without editing its parent."""
    from app.api import repository
    from app.repository.knowledge_pipeline.store import read_json
    if not repository.ai_chat_configured():
        raise HTTPException(422, "Assisted graph generation is not configured in this workspace.")
    try:
        pipeline = pipeline_for()
        inputs = read_json(pipeline.store.path(run_id) / "inputs.json")
        config = prepare_generation_config({**inputs["config"], "processing": {
            **inputs["config"].get("processing", {}), "mode": "ollama",
            "batch_chars": min(inputs["config"].get("processing", {}).get("batch_chars", 6000), 6000),
            "max_auto_batches": body.max_auto_batches}})
        metadata = await run_in_threadpool(retry_source_metadata, pipeline.store, run_id)
        title = (inputs.get("title") or "Document graph")[:165] + " — relationship-focused draft"
        run = await run_in_threadpool(pipeline.store.derive, run_id, ACTOR, config, title, idempotency_key, metadata)
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc
    if run["status"] != "created":
        return run
    return {**pipeline.store.describe(run["run_id"]), **await queue_run(run["run_id"])}


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


@router.post("/{run_id}/discover", status_code=202)
async def discover_run_sources(run_id: str):
    try:
        run = pipeline_for().request_discovery(run_id, ACTOR)
    except (ValueError, FileNotFoundError) as exc:
        raise mapped_error(exc) from exc
    return {**run, **await queue_run(run_id)}


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
