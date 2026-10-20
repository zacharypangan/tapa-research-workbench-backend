#!/usr/bin/env python3
"""Single-task outbound staging worker; standard library, no public listener.

Accepts only bounded Ollama operations and already-installed approved models.
Never downloads models, executes a received command or follows a received URL.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


STAGING_HOST = "tapa-research-workbench-backend-graph-staging.up.railway.app"
MAX_BYTES = 8_000_000
OPERATIONS = {"chat", "generate", "embed", "embeddings"}
APPROVED_MODELS = ("qwen3:8b", "qwen3-vl:8b-instruct", "qwen3-embedding:0.6b")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def request_hash(operation, body):
    return hashlib.sha256(canonical({"operation": operation, "body": body})).hexdigest()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def load_config(path, project):
    project = project.expanduser().resolve()
    if path.is_symlink() or not path.expanduser().resolve().is_relative_to(project):
        raise ValueError("Worker configuration must be a regular file inside the pilot project")
    descriptor = os.open(path.expanduser(), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_size > 16_384):
            raise ValueError("Worker configuration must be user-owned, mode 0600 and <=16 KB")
        with os.fdopen(descriptor, "rb", closefd=False) as incoming:
            config = json.load(incoming)
    finally:
        os.close(descriptor)
    if not isinstance(config, dict) or set(config) != {"backend_base_url", "worker_id", "token"}:
        raise ValueError("Worker configuration requires backend_base_url, worker_id and token only")
    origin = urlsplit(config["backend_base_url"])
    if (origin.scheme != "https" or origin.hostname != STAGING_HOST
            or origin.port not in (None, 443) or origin.username or origin.password
            or origin.query or origin.fragment or origin.path.rstrip("/") not in ("", "/api/v1")):
        raise ValueError("Only the approved Railway staging HTTPS origin is allowed")
    if not isinstance(config["worker_id"], str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", config["worker_id"]):
        raise ValueError("Worker ID must be a short slug")
    if not isinstance(config["token"], str) or len(config["token"]) < 32 or any(character.isspace() for character in config["token"]):
        raise ValueError("Worker token must be at least 32 characters without whitespace")
    config["backend_base_url"] = "https://" + STAGING_HOST + "/api/v1"
    return config


def validate_task(task, now=None):
    if not isinstance(task, dict):
        raise ValueError("Task must be an object")
    for field in ("id", "lease_token", "worker_id", "request_hash"):
        if not isinstance(task.get(field), str) or not task[field] or len(task[field]) > 200:
            raise ValueError("Task credential or identity is invalid")
    operation, body = task.get("operation"), task.get("body")
    if operation not in OPERATIONS or not isinstance(body, dict):
        raise ValueError("Operation is not allowed")
    if body.get("model") not in APPROVED_MODELS:
        raise ValueError("Model is not approved")
    if (operation in {"embed", "embeddings"}) != (body["model"] == APPROVED_MODELS[2]):
        raise ValueError("Model does not serve this operation")
    allowed = {
        "chat": {"model", "messages", "format", "options", "stream", "think", "keep_alive"},
        "generate": {"model", "prompt", "system", "format", "options", "stream", "think", "keep_alive", "images"},
        "embed": {"model", "input", "truncate", "options", "keep_alive", "dimensions"},
        "embeddings": {"model", "prompt", "options", "keep_alive"},
    }
    if set(body) - allowed[operation] or len(canonical(body)) > MAX_BYTES:
        raise ValueError("Request has unsupported fields or is too large")
    if body.get("keep_alive", 0) not in (0, "0", "0s"):
        raise ValueError("Persistent model retention is not allowed")
    options = body.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("Model options must be an object")
    if operation in {"chat", "generate"}:
        if body.get("stream") is not False or body.get("think") is not False:
            raise ValueError("Only non-streaming requests with thinking disabled are allowed")
        for field, maximum in (("num_ctx", 16384), ("num_predict", 8192)):
            value = options.get(field)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("Model context/output budget is missing or out of bounds")
        if body["model"] == APPROVED_MODELS[1] and options["num_predict"] > 4096:
            raise ValueError("Vision output budget exceeds admission limit")
    if set(options) - {"temperature", "num_ctx", "num_predict", "seed", "top_k", "top_p", "repeat_penalty", "stop"}:
        raise ValueError("Unsupported runtime option")
    if operation == "embed" and body.get("truncate") is not False:
        raise ValueError("Embedding input must not be silently truncated")
    if request_hash(operation, body) != task["request_hash"]:
        raise ValueError("Request hash mismatch")
    now = time.time() if now is None else now
    for field in ("deadline_at", "lease_until"):
        value = task.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= now:
            raise ValueError("Task deadline or lease has expired")
    if task["deadline_at"] - now > 120:
        raise ValueError("Task deadline exceeds worker admission limit")
    return operation, body


class Worker:
    def __init__(self, config, ollama_base, stop=None):
        origin = urlsplit(ollama_base)
        if (origin.scheme != "http" or origin.hostname not in ("127.0.0.1", "::1")
                or origin.username or origin.password or origin.query or origin.fragment
                or origin.path.rstrip("/") not in ("", "/api")):
            raise ValueError("Runtime must be a fixed loopback HTTP origin")
        self.config = config
        self.local_base = ollama_base.rstrip("/")
        if not self.local_base.endswith("/api"):
            self.local_base += "/api"
        self.stop = stop or threading.Event()
        self.opener = build_opener(ProxyHandler({}), NoRedirect())
        self.heartbeat_seconds = 5
        self.capabilities = None

    def http(self, url, payload=None, token=None, timeout=10, cancelled=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        data = canonical(payload) if payload is not None else None
        if data and len(data) > MAX_BYTES:
            raise ValueError("Outbound payload exceeds limit")
        request = Request(url, data=data, headers=headers)
        with self.opener.open(request, timeout=timeout) as response:
            chunks, length = [], 0
            while True:
                if self.stop.is_set() or cancelled is not None and cancelled.is_set():
                    raise InterruptedError("Task interrupted")
                chunk = response.read(min(65536, MAX_BYTES + 1 - length))
                if not chunk:
                    break
                chunks.append(chunk)
                length += len(chunk)
                if length > MAX_BYTES:
                    raise ValueError("Response exceeds limit")
            result = json.loads(b"".join(chunks))
            canonical(result)  # Reject nonfinite JSON numbers before passing results.
            if not isinstance(result, dict):
                raise ValueError("Response must be an object")
            return result

    def broker(self, operation, payload):
        return self.http(self.config["backend_base_url"] + "/model-workers/" + operation,
                         payload, self.config["token"], timeout=8)

    def inventory(self):
        response = self.http(self.local_base + "/tags", timeout=5)
        installed = {item.get("name"): item.get("digest") for item in response.get("models", []) if isinstance(item, dict)}
        if any(not isinstance(installed.get(name), str) or not installed[name] for name in APPROVED_MODELS):
            raise ValueError("All three approved models must already be installed")
        self.capabilities = {"models": list(APPROVED_MODELS),
                             "digests": {name: installed[name] for name in APPROVED_MODELS}}
        return self.capabilities

    def process_task(self, task):
        credentials = {"task_id": task.get("id"), "worker_id": self.config["worker_id"],
                       "lease_token": task.get("lease_token")}
        try:
            operation, body = validate_task(task)
            if task["worker_id"] != self.config["worker_id"]:
                raise ValueError("Task belongs to another worker")
        except (TypeError, ValueError):
            self.broker("failure", {**credentials, "category": "request_rejected", "retryable": False})
            return "request_rejected"
        finished, lost = threading.Event(), threading.Event()
        lease_until = [task["lease_until"]]

        def heartbeat():
            while not finished.wait(self.heartbeat_seconds):
                if self.stop.is_set() or time.time() >= task["deadline_at"]:
                    lost.set()
                    return
                try:
                    reply = self.broker("heartbeat", credentials)
                    lease_until[0] = min(float(reply.get("lease_until", time.time() + 15)), task["deadline_at"])
                except HTTPError as exc:
                    if exc.code in {401, 403, 404, 409, 410}:
                        lost.set()
                        return
                except (OSError, ValueError):
                    pass
                if time.time() >= lease_until[0] - 1:
                    lost.set()
                    return

        thread = threading.Thread(target=heartbeat, name="model-task-heartbeat", daemon=True)
        thread.start()
        try:
            timeout = min(75, task["deadline_at"] - time.time() - 1)
            if timeout <= 0:
                raise TimeoutError("Task deadline reached")
            result = self.http(self.local_base + "/" + operation, body,
                               timeout=timeout, cancelled=lost)
            if lost.is_set() or self.stop.is_set() or time.time() >= min(task["deadline_at"], lease_until[0]):
                return "lease_lost"
            inferred_model = operation == "embeddings" and "model" not in result
            if not inferred_model and result.get("model") != body["model"]:
                raise ValueError("Runtime returned a different model")
            if inferred_model:
                result = {**result, "model": body["model"]}
            audit = {"model_digest": self.capabilities["digests"][body["model"]]}
            if inferred_model:
                audit["response_model_inferred"] = True
            self.broker("result", {**credentials, "request_hash": task["request_hash"],
                                   "result": result, "audit": audit})
            return "succeeded"
        except HTTPError as exc:
            category, retryable = "provider_http_error", exc.code >= 500 or exc.code in {408, 429}
        except (TimeoutError, URLError):
            category, retryable = "provider_timeout", True
        except InterruptedError:
            category, retryable = "worker_error", True
        except (OSError, ValueError, KeyError):
            category, retryable = "invalid_response", False
        finally:
            finished.set()
            thread.join(timeout=9)
        if not lost.is_set() and time.time() < task["deadline_at"]:
            self.broker("failure", {**credentials, "category": category, "retryable": retryable})
        return category

    def run(self):
        self.inventory()
        failures = 0
        while not self.stop.is_set():
            try:
                reply = self.broker("claim", {"worker_id": self.config["worker_id"],
                                               "capabilities": self.capabilities})
                task = reply.get("task")
                if task is not None:
                    state = self.process_task(task)
                    # Only identity/state; no document body, result or credentials.
                    print(json.dumps({"event": "task_finished", "task_id": task.get("id"), "state": state}), flush=True)
                failures = 0
                self.stop.wait(min(5, max(1, float(reply.get("poll_after_seconds", 2)))))
            except HTTPError as exc:
                if exc.code in {401, 403}:
                    raise RuntimeError("Worker authentication was rejected") from None
                failures += 1
                self.stop.wait(min(30, 2 ** min(failures, 5)))
            except (OSError, ValueError, TypeError):
                failures += 1
                self.stop.wait(min(30, 2 ** min(failures, 5)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ollama-base", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        config = load_config(args.config, args.project)
        worker = Worker(config, args.ollama_base)
        signal.signal(signal.SIGTERM, lambda signum, frame: worker.stop.set())
        signal.signal(signal.SIGINT, lambda signum, frame: worker.stop.set())
        worker.run()
        return 0
    except Exception as exc:
        print(json.dumps({"event": "worker_stopped", "reason": type(exc).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
