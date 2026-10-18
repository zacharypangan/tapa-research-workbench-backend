"""Graph-only durable jobs on the hosted Workbench's existing persistent volume.

No Progress database or external worker is required. Each API process may run a
scheduler: an advisory execution lock prevents duplicate work on shared storage.
Queued jobs and jobs interrupted by a process restart are picked up on startup.
Completed stage checkpoints are preserved by ResearchPipeline.
Two bounded consumers let a short document proceed beside a long corpus run.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import fcntl
import logging
from pathlib import Path
import uuid

from .service import ResearchPipeline
from .store import RunBusyError, RunStore, now, read_json, write_json

LOGGER = logging.getLogger(__name__)
SHARED_WORKSPACE = "hosted-workbench"
_scheduler: GraphScheduler | None = None


def shared_store() -> RunStore:
    from app.api import repository
    return RunStore(Path(repository.STORAGE_ROOT) / "research_runs", SHARED_WORKSPACE)


def configured_pipeline(store: RunStore) -> ResearchPipeline:
    from app.api import repository
    from .processing import call_ollama
    from .runtime import collector_for_runtime, existing_collector_runtime

    async def provider(units, ontology):
        return await call_ollama(units, ontology, repository.effective_ollama_base_url(),
                                 repository.OLLAMA_RETRIEVAL_MODEL, repository.ollama_headers())

    def collector(frontier, output_dir, options):
        return collector_for_runtime(existing_collector_runtime())(frontier, output_dir, options)

    return ResearchPipeline(store, provider=provider if repository.ai_chat_configured() else None,
                            collector=collector)


@contextmanager
def execution_lock(path: Path):
    with (path / ".job.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class GraphScheduler:
    def __init__(self, store: RunStore, pipeline_factory=configured_pipeline, poll_seconds: float = 1.0):
        self.store, self.pipeline_factory = store, pipeline_factory
        self.poll_seconds = poll_seconds
        self._task: asyncio.Task | None = None
        self._wakeup = asyncio.Event()

    def enqueue(self, run_id: str) -> dict:
        path = self.store.path(run_id)
        with self.store.lock(path):
            previous = self.store.describe(run_id)
            if previous["status"] == "released":
                raise ValueError("Published graphs are immutable; create a new graph")
            job = read_json(path / ".job.json", {})
            if job.get("status") not in {"queued", "running"}:
                job = {"job_id": str(uuid.uuid4()), "run_id": run_id,
                       "status": "queued", "created_at": now(), "attempts": 0}
                write_json(path / ".job.json", job)
                self.store.state(path, "queued", previous["stage"], job_id=job["job_id"])
        return {"job_id": job["job_id"], "job_status": job["status"]}

    def wake(self):
        self._wakeup.set()

    def start(self):
        if self._task is None or self._task.done():
            self._wakeup = asyncio.Event()
            self._task = asyncio.create_task(self._work(), name="workbench-document-graphs")

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def pending(self) -> list[str]:
        queued = []
        for path in self.store.root.iterdir():
            if not path.is_dir() or path.is_symlink():
                continue
            try:
                self.store.path(path.name)
                job = read_json(path / ".job.json", {})
                if job.get("status") in {"queued", "running"}:
                    queued.append((job.get("created_at", ""), path.name))
            except (ValueError, FileNotFoundError, OSError):
                continue
        return [run_id for _, run_id in sorted(queued)]

    async def process(self, run_id: str) -> bool:
        path = self.store.path(run_id)
        with execution_lock(path) as acquired:
            if not acquired:
                return False
            with self.store.lock(path):
                job = read_json(path / ".job.json", {})
                if job.get("status") not in {"queued", "running"}:
                    return False
                job.update(status="running", started_at=now(), attempts=job.get("attempts", 0) + 1)
                write_json(path / ".job.json", job)
            try:
                await self.pipeline_factory(self.store).run(run_id)
            except RunBusyError:
                # An API operation can take the run lock between claiming a job
                # and entering the pipeline. Keep the job pending for retry.
                job.update(status="queued")
                write_json(path / ".job.json", job)
                return False
            except asyncio.CancelledError:
                job.update(status="queued", interrupted_at=now())
                write_json(path / ".job.json", job)
                previous = self.store.describe(run_id)
                self.store.state(path, "queued", previous["stage"], job_id=job["job_id"],
                                 detail="Processing paused during a restart; saved checkpoints will resume automatically")
                raise
            except Exception as exc:
                try:
                    with self.store.lock(path):
                        previous = self.store.describe(run_id)
                        if previous["status"] not in {"failed", "released"}:
                            self.store.state(path, "failed", previous["stage"], error_type=type(exc).__name__,
                                             detail="Processing could not start; saved inputs can be resumed")
                        job.update(status="failed", finished_at=now(), error_type=type(exc).__name__)
                        write_json(path / ".job.json", job)
                except RunBusyError:
                    job.update(status="queued")
                    write_json(path / ".job.json", job)
                    return False
                LOGGER.warning("Document graph job failed: run=%s error_type=%s", run_id, type(exc).__name__)
            else:
                job.update(status="completed", finished_at=now())
                write_json(path / ".job.json", job)
            return True

    async def _work(self):
        workers = [asyncio.create_task(self._consume(), name=f"document-graph-consumer-{index}")
                   for index in range(2)]
        try:
            await asyncio.gather(*workers)
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    async def _consume(self):
        while True:
            self._wakeup.clear()
            for run_id in await asyncio.to_thread(self.pending):
                try:
                    await self.process(run_id)
                except ValueError:
                    # Another API process or a researcher currently holds this run.
                    continue
            try:
                await asyncio.wait_for(self._wakeup.wait(), self.poll_seconds)
            except asyncio.TimeoutError:
                pass


def get_scheduler() -> GraphScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = GraphScheduler(shared_store())
    return _scheduler
