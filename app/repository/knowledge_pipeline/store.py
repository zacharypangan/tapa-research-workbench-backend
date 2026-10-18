"""Organization-scoped artifacts, immutable inputs and resumable run checkpoints."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import PIPELINE_VERSION


class RunBusyError(ValueError):
    """A concurrent operation owns the run; its caller may retry safely."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class RunStore:
    def __init__(self, root: Path, organization: str):
        if not organization:
            raise ValueError("A project organization is required")
        self.root = Path(root) / hashlib.sha256(organization.encode()).hexdigest()[:32]
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, run_id: str) -> Path:
        if str(uuid.UUID(run_id)) != run_id:
            raise ValueError("Invalid run ID")
        path = self.root / run_id
        if not (path / "inputs.json").exists():
            raise FileNotFoundError("Research run not found")
        return path

    @contextmanager
    def lock(self, path: Path):
        with (path / ".lock").open("a") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RunBusyError("Research run is busy; retry after its current job finishes") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def create(self, documents: list[dict], config: dict, actor: str, idempotency_key: str | None = None,
               title: str | None = None, parent_run_id: str | None = None) -> dict:
        if not 1 <= len(documents) <= 20:
            raise ValueError("Supply between 1 and 20 registered documents")
        if title is not None and (not isinstance(title, str) or not title.strip() or len(title) > 200):
            raise ValueError("Run title must contain between 1 and 200 characters")
        total = 0
        descriptors = []
        for index, document in enumerate(documents, 1):
            path = Path(document["path"])
            if not path.is_file() or path.is_symlink():
                raise ValueError("Document is not a regular file")
            total += path.stat().st_size
            if path.stat().st_size > 50 * 1024 * 1024:
                raise ValueError("A document exceeds the 50 MiB ingestion limit")
            if total > 100 * 1024 * 1024:
                raise ValueError("Run input exceeds 100 MiB")
            if path.suffix.lower() not in {".pdf", ".txt", ".md", ".json", ".docx"}:
                raise ValueError("Supported input formats: PDF, TXT, MD, JSON and DOCX")
            descriptors.append({"source_id": f"DOC_{index:03d}", "title": document.get("title", path.name),
                                "original_locator": document.get("locator", path.name),
                                "input_path": f"inputs/DOC_{index:03d}{path.suffix.lower()}",
                                "sha256": file_hash(path), "original_filename": document.get("filename", path.name),
                                "ocr_languages": document.get("ocr_languages")})
            if document.get("representation_kind") == "registered_extraction":
                descriptors[-1]["representation_kind"] = "registered_extraction"
        implementation = {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
        shared = Path(__file__).parent.parent / "document_representation.py"
        if shared.is_file():
            implementation["document_representation.py"] = file_hash(shared)
        fingerprint = digest({"documents": descriptors, "config": config, "title": title,
                              "pipeline_version": PIPELINE_VERSION,
                              "implementation_sha256": implementation, "parent_run_id": parent_run_id})
        # Serialize creation/idempotency lookup independently of per-run processing.
        with self.lock(self.root):
            index_path = self.root / "idempotency.json"
            index = read_json(index_path, {})
            key = digest(idempotency_key) if idempotency_key else None
            if key and key in index:
                prior = index[key]
                if prior["fingerprint"] != fingerprint:
                    raise ValueError("Idempotency key was already used with different inputs")
                return self.describe(prior["run_id"])
            run_id = str(uuid.uuid4())
            path = self.root / run_id
            (path / "inputs").mkdir(parents=True)
            try:
                for document, descriptor in zip(documents, descriptors):
                    target = path / descriptor["input_path"]
                    shutil.copyfile(document["path"], target)
                    if file_hash(target) != descriptor["sha256"]:
                        raise ValueError("Input changed during snapshot acquisition; create a new run")
                write_json(path / "inputs.json", {"run_id": run_id, "pipeline_version": PIPELINE_VERSION,
                           "fingerprint": fingerprint, "ontology_sha256": digest(config["ontology"]),
                           "implementation_sha256": implementation,
                           "created_at": now(), "created_by": actor, "title": title.strip() if title else None,
                           "parent_run_id": parent_run_id,
                           "documents": descriptors, "config": config})
                write_json(path / "state.json", {"run_id": run_id, "status": "created", "stage": "ingest", "updated_at": now()})
            except Exception:
                shutil.rmtree(path)
                raise
            if key:
                index[key] = {"run_id": run_id, "fingerprint": fingerprint}
                write_json(index_path, index)
        return self.describe(run_id)

    def derive(self, run_id: str, actor: str, config: dict | None = None,
               title: str | None = None, idempotency_key: str | None = None,
               source_metadata: dict | None = None) -> dict:
        """Explicit current-code retry from immutable inputs; never rewrite old evidence."""
        from .service import validate_config
        path = self.path(run_id)
        original = read_json(path / "inputs.json")
        selected = validate_config(config if config is not None else original["config"])
        documents = []
        for document in original["documents"]:
            snapshot = path / document["input_path"]
            if file_hash(snapshot) != document["sha256"]:
                raise ValueError("Original input snapshot was modified")
            metadata = (source_metadata or {}).get(document["original_locator"], {})
            documents.append({"path": snapshot, "title": document["title"], "locator": document["original_locator"],
                              "filename": metadata.get("filename") or document.get("original_filename") or (document["title"] + snapshot.suffix),
                              "ocr_languages": metadata.get("ocr_languages") or document.get("ocr_languages")})
            if document.get("representation_kind") == "registered_extraction":
                documents[-1]["representation_kind"] = "registered_extraction"
        version = {file.name: file_hash(file) for file in sorted(Path(__file__).parent.glob("*.py"))}
        shared = Path(__file__).parent.parent / "document_representation.py"
        if shared.is_file():
            version["document_representation.py"] = file_hash(shared)
        # Default retry replay is idempotent, including after a deployment/reload.
        key = idempotency_key or "derived-" + digest([run_id, selected, title, version, [{key: item.get(key) for key in ("filename", "ocr_languages")} for item in documents]])
        return self.create(documents, selected, actor, key, title or original.get("title"), parent_run_id=run_id)

    def summary(self, run_id: str) -> dict:
        path = self.path(run_id)
        state = read_json(path / "state.json")
        inputs = read_json(path / "inputs.json")
        documents = inputs["documents"]
        topic = inputs["config"].get("research_topic", "")
        title = inputs.get("title") or topic.strip() or documents[0]["title"]
        return {**state, "pipeline_version": inputs["pipeline_version"], "fingerprint": inputs["fingerprint"],
                "ontology_sha256": inputs["ontology_sha256"], "created_at": inputs["created_at"],
                "parent_run_id": inputs.get("parent_run_id"),
                "title": title, "research_topic": topic, "document_count": len(documents),
                "documents": [{"source_id": item["source_id"], "title": item["title"],
                               "locator": item["original_locator"]} for item in documents],
                "processing_mode": inputs["config"].get("processing", {}).get("mode", "manual"),
                "max_auto_batches": inputs["config"].get("processing", {}).get("max_auto_batches"),
                "has_saved_candidates": self.has_saved_candidates(path, state),
                "augmentation_enabled": inputs["config"].get("augmentation", {}).get("enabled", False),
                "max_pages": inputs["config"].get("augmentation", {}).get("max_pages", 5)}

    @staticmethod
    def has_saved_candidates(path: Path, state: dict) -> bool:
        """Small listing hint; a saved candidate is still explicitly unreviewed."""
        preferred = ([path / state["release_path"] / "graph.json"]
                     if state.get("status") == "released" and state.get("release_path") else [])
        preferred += [path / "graph_input.json", path / "validated_preview.json"]
        try:
            for artifact in preferred:
                if artifact.is_file():
                    return bool(read_json(artifact, {}).get("entities"))
            for directory in ("processed", "validated_candidates"):
                for artifact in sorted((path / directory).glob("*.json"))[:20]:
                    if read_json(artifact, {}).get("entities"):
                        return True
        except (OSError, ValueError, AttributeError):
            return False
        return False

    def list_runs(self, limit: int = 20, offset: int = 0, material_id: str | None = None) -> dict:
        if not 1 <= limit <= 100 or not 0 <= offset <= 100000:
            raise ValueError("Run listing requires a limit of 1–100 and an offset of 0–100000")
        if material_id is not None and (not isinstance(material_id, str) or not 1 <= len(material_id) <= 255 or "/" in material_id):
            raise ValueError("Material ID must contain 1–255 characters without slashes")
        entries = []
        for path in self.root.iterdir():
            if not path.is_dir() or path.is_symlink():
                continue
            try:
                summary = self.summary(path.name)
                if material_id is not None and not any(
                    document["locator"].startswith(f"workbench:material/{material_id}/file/")
                    for document in summary["documents"]
                ):
                    continue
                entries.append(summary)
            except (ValueError, FileNotFoundError):
                # A concurrent create has no visible inputs until its atomic
                # write completes; non-run storage entries are not listed.
                continue
        entries.sort(key=lambda item: (item["created_at"], item["run_id"]), reverse=True)
        return {"items": entries[offset:offset + limit], "total": len(entries), "limit": limit, "offset": offset}

    def describe(self, run_id: str) -> dict:
        path = self.path(run_id)
        return {**self.summary(run_id),
                "artifacts": sorted(p.relative_to(path).as_posix() for p in path.rglob("*")
                                    if p.is_file() and not any(part.startswith(".") for part in p.relative_to(path).parts) and not p.name.endswith(".tmp"))}

    def state(self, path: Path, status: str, stage: str, **details) -> dict:
        state = {"run_id": path.name, "status": status, "stage": stage, "updated_at": now(), **details}
        write_json(path / "state.json", state)
        return state

    def artifact(self, run_id: str, name: str) -> Path:
        root = self.path(run_id).resolve()
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file() or any(p.startswith(".") for p in Path(name).parts):
            raise FileNotFoundError("Artifact not found")
        # Only files actually indexed for this run can be downloaded.
        if name not in self.describe(run_id)["artifacts"]:
            raise FileNotFoundError("Artifact not found")
        return candidate
