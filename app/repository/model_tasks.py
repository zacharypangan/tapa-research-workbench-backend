"""Private, bounded model requests claimed by an outbound research GPU worker.

The broker does not call providers or alter prompts. A single file lock protects
claims across API processes; leases and deadlines prevent late workers from
publishing results after reassignment. No source text enters operational status.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import time
import uuid


OPERATIONS = frozenset({"chat", "generate", "embed", "embeddings"})
FAILURE_CATEGORIES = frozenset({
    "provider_timeout", "provider_unavailable", "provider_http_error", "rate_limit",
    "authentication", "model_unavailable", "invalid_response", "invalid_request",
    "provider_unreachable", "request_rejected", "result_rejected", "resource_busy", "unsupported_task",
    "worker_error", "deadline_exceeded", "lease_expired", "attempts_exhausted", "cancelled",
})
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
MAX_STORAGE_BYTES = 64 * 1024 * 1024
MIN_FREE_BYTES = 8 * 1024 * 1024
MAX_PENDING = 32
LEASE_SECONDS = 20
MAX_ATTEMPTS = 3
RETENTION_SECONDS = 300
WORKER_MAX_AGE_SECONDS = 35
_WORKER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class ModelTaskConflict(RuntimeError):
    """A task no longer belongs to this worker lease."""


class ModelTaskCapacity(RuntimeError):
    """The bounded broker cannot safely accept another write or request."""


def _json_bytes(value, maximum=MAX_PAYLOAD_BYTES):
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise ValueError("Model task payload must be finite JSON") from exc
    if len(data) > maximum:
        raise ValueError("Model task payload exceeds its byte limit")
    return data


def task_request_hash(operation: str, body: dict) -> str:
    return hashlib.sha256(_json_bytes({"operation": operation, "body": body},
                                    MAX_PAYLOAD_BYTES + 1024)).hexdigest()


def _identifier(value):
    if not isinstance(value, str) or not _WORKER_ID.fullmatch(value):
        raise ValueError("Invalid worker identifier")
    return value


def _task_id(value):
    try:
        if not isinstance(value, str) or str(uuid.UUID(value)) != value:
            raise ValueError()
    except (ValueError, AttributeError) as exc:
        raise ValueError("Invalid model task identifier") from exc
    return value


def _body(operation, body):
    if not isinstance(operation, str) or operation not in OPERATIONS or not isinstance(body, dict):
        raise ValueError("Unsupported model operation or body")
    if not isinstance(body.get("model"), str) or not 1 <= len(body["model"]) <= 200:
        raise ValueError("A model identifier is required")
    if body.get("stream", False) is not False:
        raise ValueError("Streaming requests cannot use the durable model broker")
    if operation == "chat":
        messages = body.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 200:
            raise ValueError("Chat requires bounded messages")
        for message in messages:
            if (not isinstance(message, dict)
                    or message.get("role") not in {"system", "user", "assistant", "tool"}
                    or not isinstance(message.get("content", ""), str)):
                raise ValueError("Invalid chat message")
            images = message.get("images", [])
            if not isinstance(images, list) or len(images) > 16 or any(not isinstance(v, str) for v in images):
                raise ValueError("Invalid image request")
    elif operation == "embed":
        inputs = body.get("input")
        values = [inputs] if isinstance(inputs, str) else inputs
        if not isinstance(values, list) or not 1 <= len(values) <= 64 or any(not isinstance(v, str) or not v for v in values):
            raise ValueError("Embedding requires one to 64 nonempty input strings")
    elif not isinstance(body.get("prompt"), str) or not body["prompt"]:
        raise ValueError("Generation or legacy embeddings require a prompt")
    return json.loads(_json_bytes(body))


def _vectors(vectors, expected):
    if not isinstance(vectors, list) or len(vectors) != expected:
        raise ValueError("Embedding result count does not match the request")
    dimensions = None
    for vector in vectors:
        if not isinstance(vector, list) or not 1 <= len(vector) <= 32768:
            raise ValueError("Invalid embedding dimensions")
        if dimensions is not None and len(vector) != dimensions:
            raise ValueError("Embedding dimensions differ within one result")
        dimensions = len(vector)
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or abs(v) > 1_000_000 or not math.isfinite(v) for v in vector) or not any(vector):
            raise ValueError("Embedding values must be finite, bounded, nonzero numbers")


def _result(task, result):
    if not isinstance(result, dict):
        raise ValueError("A JSON model result is required")
    data = _json_bytes(result)
    if result.get("model", task["body"]["model"]) != task["body"]["model"]:
        raise ValueError("The result model differs from the requested model")
    operation = task["operation"]
    if operation == "chat":
        message = result.get("message")
        if (result.get("done") is not True or not isinstance(message, dict)
                or not isinstance(message.get("content"), str) or not message["content"]):
            raise ValueError("Chat did not return a complete text message")
    elif operation == "generate":
        if result.get("done") is not True or not isinstance(result.get("response"), str) or not result["response"]:
            raise ValueError("Generation did not return a complete text response")
    else:
        inputs = task["body"].get("input")
        expected = len(inputs) if operation == "embed" and isinstance(inputs, list) else 1
        vectors = result.get("embeddings") if operation == "embed" else [result.get("embedding")]
        _vectors(vectors, expected)
    return json.loads(data), hashlib.sha256(data).hexdigest()


def _capabilities(value):
    if value is None:
        return {"models": [], "operations": sorted(OPERATIONS), "digests": {}}
    if not isinstance(value, dict):
        raise ValueError("Worker capabilities must be an object")
    models, operations = value.get("models", []), value.get("operations", sorted(OPERATIONS))
    digests = value.get("digests", {})
    if (not isinstance(models, list) or len(models) > 16
            or any(not isinstance(m, str) or not 1 <= len(m) <= 200 for m in models)
            or not isinstance(operations, list) or any(not isinstance(v, str) or v not in OPERATIONS for v in operations)
            or not isinstance(digests, dict) or len(digests) > 16
            or any(k not in models or not isinstance(v, str) or len(v) > 200 for k, v in digests.items())):
        raise ValueError("Invalid worker model capabilities")
    return {"models": sorted(set(models)), "operations": sorted(set(operations)), "digests": dict(digests)}


def _request_audit(audit, effective_hash):
    if audit is None:
        return {}
    allowed = {"requested_request_sha256", "effective_request_sha256", "policy_version", "effective_policy"}
    if not isinstance(audit, dict) or set(audit) - allowed:
        raise ValueError("Invalid request audit fields")
    for key in ("requested_request_sha256", "effective_request_sha256"):
        if key in audit and (not isinstance(audit[key], str) or not re.fullmatch(r"[a-f0-9]{64}", audit[key])):
            raise ValueError("Invalid request audit digest")
    if "effective_request_sha256" in audit and audit["effective_request_sha256"] != effective_hash:
        raise ValueError("Effective request audit digest does not match the task")
    if "policy_version" in audit and (not isinstance(audit["policy_version"], str) or not 1 <= len(audit["policy_version"]) <= 100):
        raise ValueError("Invalid request policy version")
    policy = audit.get("effective_policy", {})
    if not isinstance(policy, dict) or any(not isinstance(key, str) or len(key) > 100
                                         or not isinstance(value, (int, float, bool)) for key, value in policy.items()):
        raise ValueError("Effective request policy must contain numeric limits or flags")
    return json.loads(_json_bytes(audit, 16 * 1024))


class ModelTaskStore:
    def __init__(self, root, *, clock=time.time):
        self.root, self.clock = Path(root), clock
        if self.root.is_symlink():
            raise ValueError("Model task storage must not be a symbolic link")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)

    @contextmanager
    def _lock(self):
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.root / ".store.lock", flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _path(self, task_id):
        return self.root / (_task_id(task_id) + ".json")

    def _read(self, path):
        if path.is_symlink() or path.stat().st_size > MAX_PAYLOAD_BYTES + 64 * 1024:
            raise ModelTaskConflict("Invalid model task storage entry")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ModelTaskConflict("Invalid model task record")
        return value

    def _storage_bytes(self):
        return sum(path.stat().st_size for path in self.root.iterdir()
                   if path.is_file() and not path.is_symlink() and path.name != ".store.lock")

    def _save(self, path, value):
        data = _json_bytes(value, MAX_PAYLOAD_BYTES + 64 * 1024)
        if path.is_symlink():
            raise ModelTaskConflict("Invalid model task storage entry")
        previous = path.stat().st_size if path.exists() else 0
        if self._storage_bytes() - previous + len(data) > MAX_STORAGE_BYTES:
            raise ModelTaskCapacity("Model task storage capacity is reached")
        if shutil.disk_usage(self.root).free < MIN_FREE_BYTES + len(data):
            raise ModelTaskCapacity("Model task storage reserve is low")
        temporary = self.root / (".task-" + uuid.uuid4().hex + ".tmp")
        try:
            descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    def _tasks(self):
        tasks = []
        for path in self.root.glob("*.json"):
            if path.name.startswith("_"):
                continue
            _task_id(path.stem)
            tasks.append(self._read(path))
        return tasks

    def _refresh(self, task, now):
        changed = False
        if task["status"] in {"queued", "running"} and now >= task["deadline_at"]:
            task.update(status="expired", error={"category": "deadline_exceeded", "retryable": False},
                        finished_at=now, body=None, lease_token=None, lease_until=None)
            changed = True
        elif task["status"] == "running" and now >= task["lease_until"]:
            exhausted = task["attempts"] >= MAX_ATTEMPTS
            task.update(status="failed" if exhausted else "queued", worker_id=None,
                        lease_token=None, lease_until=None, available_at=now,
                        error={"category": "attempts_exhausted" if exhausted else "lease_expired",
                               "retryable": not exhausted})
            if exhausted:
                task.update(finished_at=now, body=None)
            changed = True
        if changed:
            self._save(self._path(task["id"]), task)
        return task

    def _sweep(self, now):
        for path in self.root.glob(".task-*.tmp"):
            path.unlink(missing_ok=True)
        tasks = []
        for task in self._tasks():
            task = self._refresh(task, now)
            if task["status"] not in {"queued", "running"} and now - task["finished_at"] > RETENTION_SECONDS:
                self._path(task["id"]).unlink(missing_ok=True)
            else:
                tasks.append(task)
        return tasks

    def _worker_seen(self, worker_id, now, capabilities=None):
        path = self.root / "_workers.json"
        workers = self._read(path) if path.exists() else {}
        workers = {key: row for key, row in workers.items() if now - row["last_seen"] <= RETENTION_SECONDS}
        if worker_id not in workers and len(workers) >= MAX_PENDING:
            raise ModelTaskCapacity("Model worker registration capacity is reached")
        previous = workers.get(worker_id, {})
        workers[worker_id] = {"worker_id": worker_id, "last_seen": now,
                              "capabilities": capabilities if capabilities is not None else previous.get("capabilities", _capabilities(None))}
        self._save(path, workers)

    def enqueue(self, operation: str, body: dict, timeout_seconds: float = 80,
                request_audit: dict | None = None) -> dict:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 80:
            raise ValueError("Model task timeout must be positive and at most 80 seconds")
        body = _body(operation, body)
        request_hash = task_request_hash(operation, body)
        request_audit = _request_audit(request_audit, request_hash)
        with self._lock():
            now = self.clock()
            tasks = self._sweep(now)
            if sum(task["status"] in {"queued", "running"} for task in tasks) >= MAX_PENDING:
                raise ModelTaskCapacity("Model task pending capacity is reached")
            task = {"id": str(uuid.uuid4()), "operation": operation, "body": body,
                    "request_hash": request_hash, "request_audit": request_audit, "status": "queued",
                    "request_summary": {"model": body["model"],
                        "input_count": len(body["input"]) if operation == "embed" and isinstance(body.get("input"), list) else 1,
                        "dimensions": body.get("dimensions")},
                    "created_at": now, "deadline_at": now + timeout_seconds,
                    "available_at": now, "attempts": 0, "worker_id": None,
                    "lease_token": None, "lease_until": None, "result": None, "error": None}
            self._save(self._path(task["id"]), task)
            return task

    def claim(self, worker_id: str, capabilities: dict | None = None) -> dict | None:
        worker_id, advertised = _identifier(worker_id), _capabilities(capabilities)
        with self._lock():
            now = self.clock()
            tasks = self._sweep(now)
            self._worker_seen(worker_id, now, advertised)
            eligible = lambda task: (task["operation"] in advertised["operations"] and
                                     (capabilities is None or task["body"]["model"] in advertised["models"]))
            for task in tasks:
                if task["status"] == "running" and task["worker_id"] == worker_id:
                    return task if eligible(task) else None
            for task in sorted(tasks, key=lambda task: (task["created_at"], task["id"])):
                if task["status"] == "queued" and task["available_at"] <= now and eligible(task):
                    task.update(status="running", worker_id=worker_id, lease_token=secrets.token_urlsafe(32),
                                lease_until=min(now + LEASE_SECONDS, task["deadline_at"]),
                                attempts=task["attempts"] + 1, error=None)
                    self._save(self._path(task["id"]), task)
                    return task
            return None

    def _leased(self, task_id, worker_id, lease_token, now):
        task = self._read(self._path(task_id))
        if (task["status"] != "running" or task["worker_id"] != _identifier(worker_id)
                or not isinstance(lease_token, str) or not secrets.compare_digest(task["lease_token"] or "", lease_token)
                or now >= task["deadline_at"] or now >= task["lease_until"]):
            raise ModelTaskConflict("The model task lease is stale or expired")
        return task

    def heartbeat(self, task_id: str, worker_id: str, lease_token: str) -> dict:
        with self._lock():
            now = self.clock()
            task = self._leased(task_id, worker_id, lease_token, now)
            task["lease_until"] = min(now + LEASE_SECONDS, task["deadline_at"])
            self._worker_seen(worker_id, now)
            self._save(self._path(task_id), task)
            return task

    def complete(self, task_id: str, worker_id: str, lease_token: str, result: dict,
                 audit: dict | None = None, request_hash: str | None = None) -> dict:
        with self._lock():
            now = self.clock()
            existing = self._read(self._path(task_id))
            if request_hash is not None and (not isinstance(request_hash, str) or not secrets.compare_digest(existing["request_hash"], request_hash)):
                raise ModelTaskConflict("The result request hash does not match")
            # An acknowledgement lost in transit may be retried without running
            # inference again. Only the exact successful worker/result can replay.
            if (existing["status"] == "succeeded" and existing["worker_id"] == worker_id
                    and isinstance(lease_token, str) and secrets.compare_digest(existing["lease_token"] or "", lease_token)
                    and hashlib.sha256(_json_bytes(result)).hexdigest() == existing["result_hash"]):
                return existing
            task = self._leased(task_id, worker_id, lease_token, now)
            result, digest = _result(task, result)
            if audit is not None and not isinstance(audit, dict):
                raise ValueError("Result audit must be an object")
            audit = json.loads(_json_bytes(audit or {}, 16 * 1024))
            task.update(status="succeeded", result=result, result_hash=digest,
                        result_audit=audit, finished_at=now, body=None, lease_until=None)
            self._worker_seen(worker_id, now)
            self._save(self._path(task_id), task)
            return task

    def fail(self, task_id: str, worker_id: str, lease_token: str,
             category: str, retryable: bool = False) -> dict:
        if not isinstance(category, str) or category not in FAILURE_CATEGORIES or not isinstance(retryable, bool):
            raise ValueError("Invalid safe model failure category")
        with self._lock():
            now = self.clock()
            task = self._leased(task_id, worker_id, lease_token, now)
            retry = retryable and task["attempts"] < MAX_ATTEMPTS
            task.update(status="queued" if retry else "failed", worker_id=None,
                        lease_token=None, lease_until=None,
                        available_at=min(now + 2 ** task["attempts"], task["deadline_at"]),
                        error={"category": category, "retryable": retry})
            if not retry:
                task.update(finished_at=now, body=None)
            self._worker_seen(worker_id, now)
            self._save(self._path(task_id), task)
            return task

    def get(self, task_id: str) -> dict:
        with self._lock():
            return self._refresh(self._read(self._path(task_id)), self.clock())

    def status(self) -> dict:
        with self._lock():
            now = self.clock()
            tasks = self._sweep(now)
            path = self.root / "_workers.json"
            workers = self._read(path) if path.exists() else {}
            views = [{"worker_id": row["worker_id"], "last_seen": row["last_seen"],
                      "age_seconds": max(0, now - row["last_seen"]),
                      "available": now - row["last_seen"] <= WORKER_MAX_AGE_SECONDS,
                      "models": row["capabilities"]["models"], "digests": row["capabilities"]["digests"]}
                     for row in workers.values() if now - row["last_seen"] <= RETENTION_SECONDS]
            queued = sum(task["status"] == "queued" for task in tasks)
            running = sum(task["status"] == "running" for task in tasks)
            return {"pending_count": queued + running, "queued_count": queued,
                    "running_count": running, "storage_bytes": self._storage_bytes(),
                    "workers": sorted(views, key=lambda row: row["worker_id"])}

    def available_worker(self, model: str, max_age_seconds: float = WORKER_MAX_AGE_SECONDS) -> bool:
        if isinstance(max_age_seconds, bool) or not isinstance(max_age_seconds, (int, float)) or not 0 < max_age_seconds <= 300:
            raise ValueError("Invalid worker freshness bound")
        return any(row["age_seconds"] <= max_age_seconds and model in row["models"]
                   for row in self.status()["workers"])
