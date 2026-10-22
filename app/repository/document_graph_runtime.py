"""Versioned provider selection outside the immutable research-pipeline modules."""
from __future__ import annotations

import hashlib
import errno
from pathlib import Path
import shutil

from app.repository.knowledge_pipeline import jobs
from app.repository.knowledge_pipeline.store import digest


def candidate_identity():
    from .document_graph_candidates import profile_identity
    identity = profile_identity()
    return {**identity, "version": identity["adapter_version"],
            "method": "staged_document_claims", "requires_review": True}


def execution_identity():
    from app.api import repository
    return {"model": repository.OLLAMA_RETRIEVAL_MODEL,
            "provider_sha256": digest(repository.effective_ollama_base_url().rstrip("/")),
            "runtime_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def pipeline_identity():
    """Match the files pinned by RunStore, without changing their implementation."""
    directory = Path(jobs.__file__).parent
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(directory.glob("*.py"))}
    shared = directory.parent / "document_representation.py"
    if shared.is_file():
        files[shared.name] = hashlib.sha256(shared.read_bytes()).hexdigest()
    return digest(files)


def configured_pipeline(store):
    legacy = jobs.configured_pipeline(store)
    old_provider = legacy.provider
    if old_provider is None:
        return legacy
    pause_reason = None

    async def provider(units, ontology):
        nonlocal pause_reason
        guidance = ontology.get("extraction_guidance", {})
        selected = guidance.get("candidate_provider")
        # Old immutable runs keep their original extraction protocol.
        if selected is None:
            return await old_provider(units, ontology)
        if selected != candidate_identity():
            raise ValueError("Candidate provider changed; create a new graph run to preserve its method identity")
        if guidance.get("execution_identity") != execution_identity():
            raise ValueError("Graph model routing changed; create a new graph run with the current model")
        if pause_reason:
            raise RuntimeError("Model processing paused after a provider availability failure; saved checkpoints remain resumable")
        from app.api import repository
        from .document_graph_candidates import provider as staged_provider
        storage = Path(repository.STORAGE_ROOT)
        if storage.is_dir() and shutil.disk_usage(storage).free < 16 * 1024 * 1024:
            raise OSError(errno.ENOSPC, "Graph processing paused because checkpoint storage is low")
        result = await staged_provider(units, ontology, repository.effective_ollama_base_url(),
                                       repository.OLLAMA_RETRIEVAL_MODEL, repository.ollama_headers())
        # Persist this failed batch through the existing pipeline before stopping
        # the next call. A quota outage must not cause requests for every passage.
        for attempt in result.get("provider_attempts", []):
            status = attempt.get("http_status")
            failure = attempt.get("failure_category")
            if status in {401, 403, 429, 500, 502, 503, 504} or failure in {
                "ConnectError", "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout", "RemoteProtocolError"
            }:
                pause_reason = "provider_unavailable"
                result.setdefault("provider_metadata", {})["pause_after_checkpoint"] = pause_reason
                break
        return result

    legacy.provider = provider
    return legacy


def get_scheduler():
    """Reuse the existing durable singleton, locks and restart behavior."""
    scheduler = jobs.get_scheduler()
    if scheduler.pipeline_factory is jobs.configured_pipeline:
        scheduler.pipeline_factory = configured_pipeline
    return scheduler
