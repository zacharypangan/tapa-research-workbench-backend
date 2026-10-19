"""Durable, idempotent document preparation on the existing Workbench volume."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import errno
import shutil
import logging
from pathlib import Path
import uuid


LOG = logging.getLogger(__name__)
PIPELINE_VERSION = "document-preparation-v1"
STAGES = ("extract", "graph", "text_index", "image_index")
_scheduler = None


class ProcessingStageError(RuntimeError):
    """Stage failure that distinguishes a retry from a configuration correction."""
    def __init__(self, message, *, retryable=True):
        super().__init__(message)
        self.retryable = retryable


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    # Fail before another provider pass when its checkpoint cannot be persisted.
    if shutil.disk_usage(path.parent).free < 8 * 1024 * 1024:
        raise OSError(errno.ENOSPC, "Storage is low; checkpoint processing is paused until space is available")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        # A failed write/replace must preserve the last complete checkpoint and
        # must not accumulate orphan files while a full volume is being repaired.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            LOG.warning("Processing checkpoint temporary file cleanup failed")


@contextmanager
def lock(path, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def material_inputs(material_id):
    from app.api import repository as repo
    from .document_representation import ocr_runtime
    with repo.get_connection() as connection:
        material = dict(repo.ensure_material(connection, material_id))
        files = [dict(row) for row in connection.execute(
            "SELECT * FROM files WHERE material_id=? ORDER BY id", (material_id,))]
    documents = []
    for row in files:
        path = Path(row["stored_path"])
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for piece in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(piece)
        document = {"file_id": row["id"], "sha256": sha.hexdigest(), "filename": row["original_filename"]}
        if Path(row["original_filename"]).suffix.lower() == ".pdf":
            document["ocr_runtime"] = ocr_runtime(material.get("language"), row["original_filename"])
        documents.append(document)
    return {"material_id": material_id, "files": documents,
            "source_url": material.get("source_url"), "language": material.get("language")}


class ProcessingScheduler:
    def __init__(self, root=None, workers=2, stage_runner=None):
        if root is None:
            from app.api import repository
            root = Path(repository.STORAGE_ROOT) / "processing_jobs"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.workers = max(1, min(workers, 4))
        self.stage_runner = stage_runner or self.run_stage
        self._task = None
        self._wake = asyncio.Event()

    def path(self, job_id):
        return self.root / (str(uuid.UUID(job_id)) + ".json")

    def get(self, job_id):
        return json.loads(self.path(job_id).read_text(encoding="utf-8"))

    def update(self, job_id, **values):
        path = self.path(job_id)
        with lock(path.with_suffix(".state.lock")):
            job = self.get(job_id)
            job.update(values, updated_at=stamp())
            save(path, job)
        return job

    def enqueue(self, material_id, stages=STAGES, force=False):
        from app.api import repository
        if not stages or any(stage not in STAGES for stage in stages):
            raise ValueError("Choose extract, graph, text_index or image_index stages")
        inputs = material_inputs(material_id)
        from .document_representation import REPRESENTATION_VERSION, representation_signature
        from .knowledge_pipeline.processing import PROMPT_VERSION
        from .image_descriptions import PROMPT_VERSION as DESCRIPTION_VERSION
        index_key = getattr(repository, "active_embedding_index_key", repository.active_embedding_model_name)()
        fingerprint = digest([inputs, PIPELINE_VERSION, REPRESENTATION_VERSION, representation_signature(), PROMPT_VERSION,
                              DESCRIPTION_VERSION, repository.OLLAMA_VISION_MODEL,
                              index_key, list(stages)])
        job_id = str(uuid.uuid4()) if force else str(uuid.uuid5(uuid.NAMESPACE_URL, fingerprint))
        path = self.path(job_id)
        with lock(path.with_suffix(".state.lock")):
            if path.exists():
                return self.get(job_id)
            job = {"job_id": job_id, "material_id": material_id, "input_hash": fingerprint,
                   "inputs": inputs, "pipeline_version": PIPELINE_VERSION, "status": "queued",
                   "stage": "queued", "stages": list(stages), "completed_stages": {},
                   "created_at": stamp(), "updated_at": stamp(), "attempts": 0}
            save(path, job)
        return job

    def list_for(self, material_id):
        jobs = []
        for path in self.root.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job.get("material_id") == material_id:
                    jobs.append(job)
            except (ValueError, OSError):
                continue
        return sorted(jobs, key=lambda job: job["created_at"], reverse=True)

    def resolve(self, job_id):
        seen = set()
        original = self.get(job_id)
        job = original
        for _ in range(20):
            if job["job_id"] in seen:
                raise ValueError("Preparation replacement chain contains a cycle")
            seen.add(job["job_id"])
            if job["status"] != "superseded" or not job.get("replacement_job_id"):
                return job
            job = self.get(job["replacement_job_id"])
            if job.get("material_id") != original.get("material_id"):
                raise ValueError("Preparation replacement belongs to another source")
        raise ValueError("Preparation replacement chain exceeds limit")

    def current_for(self, material_id):
        jobs = self.list_for(material_id)
        return self.resolve(jobs[0]["job_id"]) if jobs else None

    def enqueue_group(self, material_ids, stages, force=False):
        children = [self.enqueue(material_id, stages, force)["job_id"] for material_id in material_ids]
        job_id = str(uuid.uuid5(uuid.NAMESPACE_URL, digest(children)))
        path = self.path(job_id)
        with lock(path.with_suffix(".state.lock")):
            if not path.exists():
                save(path, {"job_id": job_id, "kind": "group", "child_job_ids": children,
                            "status": "queued", "stage": "image_index", "created_at": stamp(), "updated_at": stamp()})
        self.wake()
        return self.get(job_id)

    def retry(self, job_id):
        job = self.resolve(job_id)
        if job["status"] in {"running", "queued"}:
            return job
        if job.get("kind") == "group":
            for child in job["child_job_ids"]:
                if self.resolve(child)["status"] == "error":
                    self.retry(child)
        elif "image_index" in job.get("stages", []):
            from app.api.repository import clear_image_description_failure_guard
            clear_image_description_failure_guard()
        changes = {"completed_stages": {}} if job["status"] == "done" else {}
        job = self.update(job["job_id"], status="queued", attempts=0, error=None,
                          stage_errors={}, retryable=None, detail="Retrying unfinished stages", **changes)
        self.wake()
        return job

    def wake(self):
        self._wake.set()

    def start(self):
        if self._task is None or self._task.done():
            self._wake = asyncio.Event()
            self._task = asyncio.create_task(self.work(), name="workbench-document-preparation")

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def bootstrap(self):
        from app.api import repository
        with repository.get_connection() as connection:
            ids = [row[0] for row in connection.execute(
                "SELECT id FROM materials WHERE EXISTS (SELECT 1 FROM files WHERE files.material_id=materials.id) OR trim(COALESCE(source_url, '')) != ''")]
        for material_id in ids:
            try:
                self.enqueue(material_id)
            except Exception:
                LOG.warning("Document preparation could not queue material=%s", material_id)

    def pending(self):
        result = []
        for path in self.root.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job["status"] in {"queued", "running"}:
                    result.append(job)
            except (ValueError, KeyError, OSError):
                continue
        return sorted(result, key=lambda job: job["created_at"])

    async def process(self, job_id):
        job = self.get(job_id)
        if job.get("kind") == "group":
            children = [self.resolve(child_id) for child_id in job["child_job_ids"]]
            statuses = [child["status"] for child in children]
            status = "error" if any(value in {"error", "superseded"} for value in statuses) else "done" if all(value == "done" for value in statuses) else "queued"
            self.update(job_id, status=status, result={"completed": statuses.count("done"), "total": len(children)})
            return True
        with lock(self.root / ("material-" + digest(job["material_id"]) + ".lock"), blocking=False) as acquired:
            if not acquired:
                return False
            job = self.get(job_id)
            if job["status"] not in {"queued", "running"}:
                return False
            completed = job["completed_stages"]
            job = self.update(job_id, status="running", attempts=job["attempts"] + 1, started_at=stamp())
            try:
                # A later upload supersedes this plan before any stage starts.
                current = await asyncio.to_thread(material_inputs, job["material_id"])
                if current != job["inputs"]:
                    replacement = await asyncio.to_thread(self.enqueue, job["material_id"], job["stages"])
                    self.update(job_id, status="superseded", replacement_job_id=replacement["job_id"])
                    return True
                # Retry only unfinished stages, retaining their last errors until each
                # succeeds. Never leave a resolved failure in the readiness response.
                stage_errors = {key: value for key, value in job.get("stage_errors", {}).items()
                                if key not in completed}
                for stage in job["stages"]:
                    if stage in completed:
                        continue
                    if job["attempts"] > 1 and stage_errors.get(stage, {}).get("retryable") is False:
                        # Other transient stages may retry; this one needs a user
                        # correction/explicit retry instead of another identical call.
                        continue
                    if await asyncio.to_thread(material_inputs, job["material_id"]) != job["inputs"]:
                        replacement = await asyncio.to_thread(self.enqueue, job["material_id"], job["stages"])
                        self.update(job_id, status="superseded", replacement_job_id=replacement["job_id"])
                        return True
                    self.update(job_id, stage=stage)
                    task = asyncio.create_task(self.stage_runner(stage, job))
                    try:
                        result = await asyncio.shield(task)
                    except asyncio.CancelledError:
                        # Keep the material lock until file-writing work actually stops.
                        try:
                            await task
                        except Exception:
                            pass
                        raise
                    except Exception as exc:
                        stage_errors[stage] = {"error": type(exc).__name__, "detail": str(exc)[:500],
                                               "retryable": getattr(exc, "retryable", True)}
                        self.update(job_id, stage_errors=stage_errors)
                        if stage == "extract":
                            raise
                        continue
                    completed[stage] = {"finished_at": stamp(), "result": result}
                    stage_errors.pop(stage, None)
                    self.update(job_id, completed_stages=completed, stage_errors=stage_errors)
                # Source changes take precedence over errors from the obsolete input,
                # including a failure in the final stage of the old plan.
                if await asyncio.to_thread(material_inputs, job["material_id"]) != job["inputs"]:
                    replacement = await asyncio.to_thread(self.enqueue, job["material_id"], job["stages"])
                    self.update(job_id, status="superseded", replacement_job_id=replacement["job_id"])
                elif stage_errors:
                    retryable = any(error.get("retryable", True) for error in stage_errors.values())
                    raise ProcessingStageError("Stages need attention: " + ", ".join(stage_errors), retryable=retryable)
                else:
                    self.update(job_id, status="done", stage="complete", finished_at=stamp(),
                                error=None, detail=None, stage_errors={}, retryable=None)
            except asyncio.CancelledError:
                self.update(job_id, status="queued", detail="Restart interrupted preparation; completed stages are saved")
                raise
            except Exception as exc:
                # Retry transient provider failures using already saved per-item outputs.
                retryable = getattr(exc, "retryable", True)
                status = "queued" if retryable and job["attempts"] < 3 else "error"
                self.update(job_id, status=status, error=type(exc).__name__, detail=str(exc)[:500], retryable=retryable)
                LOG.warning("Document preparation failed: job=%s stage=%s error=%s", job_id, stage if 'stage' in locals() else 'inputs', type(exc).__name__)
                if status == "queued":
                    await asyncio.sleep(min(2 ** job["attempts"], 8))
        return True

    async def work(self):
        await asyncio.to_thread(self.bootstrap)
        semaphore = asyncio.Semaphore(self.workers)

        async def bounded(job):
            async with semaphore:
                try:
                    await self.process(job["job_id"])
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    LOG.warning("Preparation job could not be read: job=%s error=%s", job["job_id"], type(exc).__name__)

        while True:
            self._wake.clear()
            await asyncio.gather(*(bounded(job) for job in await asyncio.to_thread(self.pending)))
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=2)
            except asyncio.TimeoutError:
                pass

    async def run_stage(self, stage, job):
        from app.api import repository as repo
        from .schemas import ExtractRequest
        material_id = job["material_id"]
        if stage == "extract":
            return await repo.extract_material_text(material_id, None,
                ExtractRequest(include_links=bool(job["inputs"].get("source_url")), max_link_depth=0,
                               max_link_pages=1, max_segments=5000), force=True)
        if stage == "graph":
            from .knowledge_pipeline.jobs import get_scheduler
            from .knowledge_pipeline.service import validate_config
            from .knowledge_pipeline.ontologies import ontology_template
            from app.api.research_runs import DocumentRef, resolve_documents
            with repo.get_connection() as connection:
                material = dict(repo.ensure_material(connection, material_id))
                files = connection.execute("SELECT * FROM files WHERE material_id=? ORDER BY id", (material_id,)).fetchall()
            refs = [DocumentRef(material_id=material_id, file_id=row["id"]) for row in files
                    if Path(row["original_filename"]).suffix.lower() in {".pdf", ".txt", ".md", ".json", ".docx"}]
            if not refs:
                return {"status": "skipped", "reason": "No supported graph documents"}
            if len(refs) > 20:
                return {"status": "blocked", "reason": "Select up to 20 files for a graph"}
            documents = await asyncio.to_thread(resolve_documents, refs)
            config = validate_config({"ontology": ontology_template("general"),
                                      "research_topic": material.get("collection") or "documented research evidence",
                                      "processing": {"mode": "ollama" if repo.ai_chat_configured() else "manual",
                                                     "max_auto_batches": 2},
                                      "augmentation": {"enabled": False}})
            scheduler = get_scheduler()
            run = await asyncio.to_thread(scheduler.store.create, documents, config, "workbench-researcher",
                                          "auto-prepare:" + job["input_hash"], material["title"])
            if run["status"] == "created":
                await asyncio.to_thread(scheduler.enqueue, run["run_id"])
                scheduler.wake()
            return {"run_id": run["run_id"], "status": run["status"], "review_required": True}
        if not repo.ai_embedding_configured():
            raise ProcessingStageError("Embedding provider is not configured; extracted text and graph drafts remain available", retryable=False)
        operation = repo.index_missing_segment_embeddings if stage == "text_index" else repo.index_missing_image_embeddings
        progress = self.get(job["job_id"]).get("stage_progress", {}).get(stage, {})
        total, skipped = progress.get("indexed_count", 0), progress.get("skipped_count", 0)
        failed_ids, failed_retryability, error_counts = set(), {}, {}
        for batch in range(100):
            with repo.get_connection() as connection:
                result = await operation(connection, material_id=material_id, limit=200, force=False, skip_ids=failed_ids)
            total += result.get("indexed_count", 0)
            skipped += result.get("skipped_count", 0)
            failed_ids.update(result.get("failed_ids", []))
            for outcome in result.get("outcomes", []):
                if outcome.get("status") != "failed":
                    continue
                item_id = outcome.get("id")
                if item_id is not None:
                    failed_retryability[item_id] = outcome.get("retryable", True) is not False
                category = outcome.get("error_category", "processing_error")
                http_status = outcome.get("http_status")
                key = (category, http_status, outcome.get("retryable", True) is not False)
                error_counts[key] = error_counts.get(key, 0) + 1
            with (self.root / (job["job_id"] + "." + stage + ".outcomes.jsonl")).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"at": stamp(), "outcomes": result.get("outcomes", [])}, ensure_ascii=False) + "\n")
            stage_progress = self.get(job["job_id"]).get("stage_progress", {})
            stage_progress[stage] = {"indexed_count": total, "skipped_count": skipped,
                                     "pending_count": result.get("pending_count"),
                                     "failed_count": len(failed_ids),
                                     "failed_ids": sorted(failed_ids, key=str),
                                     "error_summary": [{"error_category": category, "http_status": status,
                                                        "retryable": retryable, "count": count}
                                                       for (category, status, retryable), count in error_counts.items()],
                                     "batch": batch + 1, "last_outcomes": result.get("outcomes", [])[-20:]}
            self.update(job["job_id"], stage_progress=stage_progress)
            if failed_ids and (not result.get("outcomes") or result.get("pending_count", 0) <= len(failed_ids)):
                raise ProcessingStageError(f"{stage}: {len(failed_ids)} item(s) need retry or correction; other eligible items were processed",
                                           retryable=any(failed_retryability.get(item, True) for item in failed_ids))
            if not result.get("pending_count", result.get("indexed_count", 0)):
                return {**result, "total_indexed_count": total}
            if result.get("indexed_count", 0) == 0 and result.get("skipped_count", 0) == 0 and not result.get("failed_count", 0):
                raise RuntimeError(f"{stage}: processing made no progress; inspect item outcomes")
        raise RuntimeError(f"{stage}: batch budget reached; successful checkpoints are preserved")


def get_processing_scheduler():
    global _scheduler
    from app.api import repository
    root = Path(repository.STORAGE_ROOT) / "processing_jobs"
    if _scheduler is None or _scheduler.root != root:
        _scheduler = ProcessingScheduler(root)
    return _scheduler


def enqueue_material(material_id, stages=STAGES, force=False):
    scheduler = get_processing_scheduler()
    result = scheduler.enqueue(material_id, stages, force)
    scheduler.wake()
    return result
