"""Bounded authenticated Ollama gateway for outbound private GPU workers.

Disabled unless its two independent secrets are configured. Document payloads
remain inside authenticated task envelopes; status and errors contain no source
text, provider bodies, credentials, or server addresses.
"""

from __future__ import annotations

import asyncio
import base64
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import time

from fastapi import APIRouter, HTTPException, Request

from app.repository.model_tasks import ModelTaskCapacity, ModelTaskConflict, ModelTaskStore
from app.repository.settings import STORAGE_ROOT

router = APIRouter(tags=["private-model-workers"])
MODELS = {"qwen3:8b", "qwen3-vl:8b-instruct", "qwen3-embedding:0.6b"}
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
VISION_MODEL = "qwen3-vl:8b-instruct"
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARACTERS = 256 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024
POLICY_VERSION = "minpaku-bounded-v1"
POLL_SECONDS = 0.25
FAILURE_CATEGORIES = {
    "provider_unreachable", "provider_http_error", "provider_timeout", "invalid_response",
    "request_rejected", "result_rejected", "worker_error", "deadline_exceeded",
    "model_unavailable", "resource_busy", "unsupported_task",
}
DIGEST = re.compile(r"(?:sha256:)?[a-f0-9]{64}\Z")
WORKER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def _error(status: int, detail: str):
    raise HTTPException(status, detail)


def _authorize(request: Request, name: str):
    tokens = {key: os.getenv(key, "").strip() for key in ("MODEL_GATEWAY_TOKEN", "MODEL_WORKER_TOKEN")}
    expected = tokens[name]
    if any(len(value) < 32 for value in tokens.values()) or secrets.compare_digest(
            tokens["MODEL_GATEWAY_TOKEN"].encode(), tokens["MODEL_WORKER_TOKEN"].encode()):
        _error(503, "This private model service is disabled.")
    scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(expected.encode(), supplied.encode()):
        raise HTTPException(401, "Private model service authorization required.", headers={"WWW-Authenticate": "Bearer"})


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(operation, body):
    return hashlib.sha256(_canonical({"operation": operation, "body": body}).encode()).hexdigest()


async def _read_json(request: Request):
    length = request.headers.get("content-length")
    if length is not None:
        try:
            if int(length) < 0:
                raise ValueError
            if int(length) > MAX_BODY_BYTES:
                _error(413, "Model request exceeds the size limit.")
        except ValueError:
            _error(400, "Invalid model request length.")
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > MAX_BODY_BYTES:
            _error(413, "Model request exceeds the size limit.")
        raw.extend(chunk)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    def reject_constant(_):
        raise ValueError

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
        if not isinstance(value, dict):
            raise ValueError
        _canonical(value)
        return value
    except (ValueError, TypeError, RecursionError, UnicodeError):
        _error(400, "Model request must be a finite JSON object with unique keys.")


@lru_cache(maxsize=4)
def _store_at(root):
    return ModelTaskStore(root)


def get_model_task_store():
    return _store_at(str(Path(STORAGE_ROOT) / "model-tasks"))


def _string(value, maximum=MAX_TEXT_CHARACTERS, *, empty=False):
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        _error(422, "Model request contains invalid text fields.")
    return value


def _images(values):
    if not isinstance(values, list) or len(values) > 4:
        _error(422, "Model request contains too many images.")
    total = 0
    for value in values:
        if not isinstance(value, str) or len(value) > (MAX_IMAGE_BYTES * 4 // 3 + 4):
            _error(422, "Model image exceeds the input limit.")
        try:
            decoded = base64.b64decode(value, validate=True)
        except (ValueError, TypeError):
            _error(422, "Model images must be encoded image bytes.")
        if not decoded or len(decoded) > MAX_IMAGE_BYTES:
            _error(422, "Model image exceeds the input limit.")
        if not (decoded.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a"))
                or (decoded.startswith(b"RIFF") and decoded[8:12] == b"WEBP")):
            _error(422, "Model image format is not supported.")
        total += len(decoded)
    return total


def _options(options):
    if not isinstance(options, dict) or set(options) - {"temperature", "top_p", "top_k", "seed", "stop", "num_ctx", "num_predict"}:
        _error(422, "Model options are not supported.")
    for key, value in options.items():
        if key == "stop":
            if not isinstance(value, list) or len(value) > 8 or any(not isinstance(item, str) or len(item) > 128 for item in value):
                _error(422, "Model stop options are invalid.")
        elif type(value) not in (int, float) or abs(value) > 10**18 or not math.isfinite(value):
            _error(422, "Model options must be bounded finite numbers.")
        elif key in {"num_ctx", "num_predict", "top_k", "seed"} and type(value) is not int:
            _error(422, "Model integer options are invalid.")
        elif key == "temperature" and not 0 <= value <= 2:
            _error(422, "Model temperature is outside the allowed range.")
        elif key == "top_p" and not 0 <= value <= 1:
            _error(422, "Model probability option is outside the allowed range.")
        elif key == "top_k" and not 0 <= value <= 100:
            _error(422, "Model top-k option is outside the allowed range.")
        elif key == "seed" and not -1 <= value <= 2**31 - 1:
            _error(422, "Model seed is outside the allowed range.")
        elif key in {"num_ctx", "num_predict"} and value < 1:
            _error(422, "Model token options must be positive.")


def _schema(value, depth=0):
    if depth > 30:
        _error(422, "Model output schema is too deeply nested.")
    if isinstance(value, dict):
        reference = value.get("$ref")
        if reference is not None and (not isinstance(reference, str) or not reference.startswith("#/")):
            _error(422, "External model schema references are not permitted.")
        for item in value.values():
            _schema(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _schema(item, depth + 1)


def _bound_graph_schema(body):
    schema = body.get("format")
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if not isinstance(properties, dict) or not {"entities", "assertions"} <= properties.keys():
        return False
    for key in ("entities", "assertions"):
        item = properties[key]
        try:
            if item.get("type") != "array":
                raise ValueError
            item["maxItems"] = min(item.get("maxItems", 6), 6)
            evidence = item["items"]["properties"]["evidence"]
            evidence["maxItems"] = min(evidence.get("maxItems", 2), 2)
            choices = evidence["items"].get("oneOf", [evidence["items"]])
            selected = next(choice for choice in choices if set(choice.get("required", [])) == {"unit_id", "span_id"})
            evidence["items"] = copy.deepcopy(selected)
        except (ValueError, TypeError, KeyError, AttributeError, StopIteration):
            _error(422, "Graph extraction schema must support registered span evidence.")
    guidance = ("\nBounded model gateway policy: return at most six salient entities and six supported assertions, "
                "with at most two registered span IDs per item. Select only evidence needed to support each item. "
                "Copy label_original verbatim from its selected source span, preserving OCR spaces and glyphs; "
                "put readable translations in label_en. Never normalize a source label. "
                "Do not add weak candidates to fill the allowance. Return one complete JSON object.")
    if "messages" in body:
        message = next((item for item in reversed(body["messages"]) if item["role"] == "user"), None)
        if message is None:
            _error(422, "Graph extraction requires a user evidence message.")
        message["content"] += guidance
    else:
        body["prompt"] += guidance
    return True


def prepare_gateway_request(operation, original):
    """Validate first, then deep-copy and narrow only the effective request."""
    if operation not in {"chat", "generate", "embed", "embeddings"}:
        _error(404, "Model operation is not supported.")
    model = original.get("model")
    if not isinstance(model, str) or model not in MODELS or ((operation in {"embed", "embeddings"}) != (model == EMBEDDING_MODEL)):
        _error(422, "Model is not approved for this operation.")
    if "stream" in original and original["stream"] is not False:
        _error(422, "Only non-streaming model requests are supported.")
    common = {"model", "stream", "keep_alive", "options"}
    allowed = common | ({"messages", "think", "format"} if operation == "chat" else
                        {"prompt", "system", "images", "think", "format"} if operation == "generate" else
                        {"input", "truncate", "dimensions"} if operation == "embed" else {"prompt"})
    if set(original) - allowed:
        _error(422, "Model request contains unsupported fields.")
    _options(original.get("options", {}))
    image_count, image_bytes, text_size = 0, 0, 0
    if operation == "chat":
        messages = original.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
            _error(422, "Model messages must be a bounded nonempty list.")
        for item in messages:
            if not isinstance(item, dict) or set(item) - {"role", "content", "images"} or item.get("role") not in {"system", "user", "assistant"}:
                _error(422, "Model message roles or fields are invalid.")
            text_size += len(_string(item.get("content"), empty=True))
            if "images" in item:
                image_bytes += _images(item["images"])
                image_count += len(item["images"])
    elif operation == "embed":
        values = original.get("input")
        values = [values] if isinstance(values, str) else values
        if not isinstance(values, list) or not 1 <= len(values) <= 64:
            _error(422, "Embedding input must contain one to 64 text items.")
        text_size = sum(len(_string(item, 32768)) for item in values)
        if "dimensions" in original and (type(original["dimensions"]) is not int or not 1 <= original["dimensions"] <= 8192):
            _error(422, "Embedding dimensions are outside the allowed range.")
        if "truncate" in original and type(original["truncate"]) is not bool:
            _error(422, "Embedding truncation must be boolean.")
    else:
        text_size = len(_string(original.get("prompt")))
        if "system" in original:
            text_size += len(_string(original["system"], empty=True))
        if "images" in original:
            image_bytes, image_count = _images(original["images"]), len(original["images"])
    if text_size > MAX_TEXT_CHARACTERS or image_count > 4 or image_bytes > 6 * 1024 * 1024:
        _error(422, "Model request exceeds its text or image budget.")
    if image_count and model != VISION_MODEL:
        _error(422, "Images require the approved vision model.")
    output_format = original.get("format")
    if output_format is not None:
        if output_format != "json" and not isinstance(output_format, dict):
            _error(422, "Model output format is not supported.")
        if len(_canonical(output_format)) > 128 * 1024:
            _error(422, "Model output schema exceeds its budget.")
        _schema(output_format)
    body = copy.deepcopy(original)
    body["keep_alive"] = 0
    if operation in {"chat", "generate"}:
        body["stream"] = False
    else:
        body.pop("stream", None)
    options = body.setdefault("options", {})
    options["num_ctx"] = 16384
    graph = False
    if operation in {"chat", "generate"}:
        body["think"] = False
        options["num_predict"] = min(options.get("num_predict", 4096), 4096)
        graph = _bound_graph_schema(body)
    if operation == "embed":
        body["truncate"] = False
    timeout = 35 if model == EMBEDDING_MODEL else 50 if model == VISION_MODEL else 80
    audit = {"policy_version": POLICY_VERSION, "requested_request_sha256": _digest(operation, original),
             "effective_request_sha256": _digest(operation, body), "effective_policy": {
                 "num_ctx": 16384, "keep_alive": 0, "stream": False,
                 **({"think": False, "num_predict": options["num_predict"]} if operation in {"chat", "generate"} else {"truncate": False}),
                 **({"max_entities": 6, "max_assertions": 6, "max_evidence_spans": 2} if graph else {}),
             }}
    return body, timeout, audit


def _validated_result(task, result):
    summary = task.get("request_summary") or {"model": task["body"]["model"],
        "input_count": len(task["body"]["input"]) if task["operation"] == "embed" and isinstance(task["body"].get("input"), list) else 1,
        "dimensions": task["body"].get("dimensions")}
    if not isinstance(result, dict) or result.get("model") != summary["model"] or "error" in result:
        _error(422, "Worker model result is invalid.")
    if len(_canonical(result).encode()) > MAX_BODY_BYTES:
        _error(413, "Worker model result exceeds its budget.")
    operation = task["operation"]
    safe = {key: result[key] for key in ("model", "created_at", "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration") if key in result}
    for key in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"):
        if key in safe and (type(safe[key]) is not int or not 0 <= safe[key] <= 10**18):
            _error(422, "Worker model metrics are invalid.")
    if "created_at" in safe:
        _string(safe["created_at"], 100)
    if operation in {"embed", "embeddings"}:
        vectors = result.get("embeddings") if operation == "embed" else [result.get("embedding")]
        count = summary["input_count"]
        if not isinstance(vectors, list) or len(vectors) != count:
            _error(422, "Worker embedding count does not match the request.")
        dimensions = summary["dimensions"]
        for vector in vectors:
            if not isinstance(vector, list) or not 1 <= len(vector) <= 8192:
                _error(422, "Worker embedding dimensions are invalid.")
            dimensions = len(vector) if dimensions is None else dimensions
            if len(vector) != dimensions or any(type(value) not in (int, float) or abs(value) > 10**18 or not math.isfinite(value) or abs(value) > 10**6 for value in vector) or not any(vector):
                _error(422, "Worker embedding values are invalid.")
        safe["embeddings" if operation == "embed" else "embedding"] = vectors if operation == "embed" else vectors[0]
    else:
        if result.get("done") is not True or result.get("done_reason") in {"length", "max_tokens"}:
            _error(422, "Worker model output is incomplete.")
        safe.update(done=True, done_reason=result.get("done_reason", "stop"))
        if safe["done_reason"] not in {"stop"}:
            _error(422, "Worker model completion reason is invalid.")
        if operation == "chat":
            message = result.get("message")
            if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
                _error(422, "Worker assistant message is invalid.")
            safe["message"] = {"role": "assistant", "content": _string(message.get("content"), 2 * 1024 * 1024)}
        else:
            safe["response"] = _string(result.get("response"), 2 * 1024 * 1024)
    return safe


def _worker_fields(body):
    worker = body.get("worker_id")
    if not isinstance(worker, str) or not WORKER_ID.fullmatch(worker):
        _error(422, "Worker identity is invalid.")
    return worker


def _lease_fields(body):
    worker = _worker_fields(body)
    task_id, lease = body.get("task_id"), body.get("lease_token")
    if not isinstance(task_id, str) or not re.fullmatch(r"[a-f0-9-]{32,36}", task_id) or not isinstance(lease, str) or not 16 <= len(lease) <= 256:
        _error(422, "Worker lease identity is invalid.")
    return task_id, worker, lease


async def _broker(method, *args, **kwargs):
    try:
        return await asyncio.to_thread(method, *args, **kwargs)
    except ModelTaskCapacity:
        _error(503, "Private model task capacity is temporarily exhausted.")
    except ModelTaskConflict:
        _error(409, "Worker task lease is stale or no longer available.")
    except FileNotFoundError:
        _error(404, "Private model task was not found.")
    except (ValueError, TypeError):
        _error(422, "Private model task request is invalid.")
    except OSError:
        _error(503, "Private model task storage is temporarily unavailable.")


@router.post("/model-gateway/api/{operation}")
async def model_gateway(operation: str, request: Request):
    _authorize(request, "MODEL_GATEWAY_TOKEN")
    original = await _read_json(request)
    body, timeout, audit = prepare_gateway_request(operation, original)
    store = get_model_task_store()
    if not await _broker(store.available_worker, body["model"], max_age_seconds=35):
        _error(503, "No recent private model worker is available for this model.")
    task = await _broker(store.enqueue, operation, body, timeout_seconds=timeout, request_audit=audit)
    deadline = time.monotonic() + timeout
    while True:
        current = await _broker(store.get, task["id"])
        if current["status"] == "succeeded":
            return {**current["result"], "workbench_model_audit": {"task_id": task["id"], **current.get("result_audit", audit)}}
        if current["status"] == "failed":
            _error(502, "Private model task failed; provider output was not returned.")
        if current["status"] == "expired" or time.monotonic() >= deadline:
            _error(504, "Private model task exceeded its bounded deadline.")
        await asyncio.sleep(POLL_SECONDS)


@router.post("/model-workers/claim")
async def claim_model_task(request: Request):
    _authorize(request, "MODEL_WORKER_TOKEN")
    body = await _read_json(request)
    if set(body) != {"worker_id", "capabilities"}:
        _error(422, "Worker claim fields are invalid.")
    worker = _worker_fields(body)
    capabilities = body.get("capabilities")
    if not isinstance(capabilities, dict) or set(capabilities) - {"models", "digests"}:
        _error(422, "Worker capabilities are invalid.")
    models, digests = capabilities.get("models"), capabilities.get("digests", {})
    if not isinstance(models, list) or not 1 <= len(models) <= 3 or any(not isinstance(model, str) or model not in MODELS for model in models) or len(set(models)) != len(models):
        _error(422, "Worker models are outside the approved allowlist.")
    if not isinstance(digests, dict) or set(digests) - set(models) or any(not isinstance(value, str) or not DIGEST.fullmatch(value) for value in digests.values()):
        _error(422, "Worker model digests are invalid.")
    task = await _broker(get_model_task_store().claim, worker, capabilities)
    return {"task": task, "poll_after_seconds": 2}


@router.post("/model-workers/heartbeat")
async def heartbeat_model_task(request: Request):
    _authorize(request, "MODEL_WORKER_TOKEN")
    body = await _read_json(request)
    if set(body) != {"task_id", "worker_id", "lease_token"}:
        _error(422, "Worker heartbeat fields are invalid.")
    task = await _broker(get_model_task_store().heartbeat, *_lease_fields(body))
    return {key: task[key] for key in ("status", "lease_until", "deadline_at")}


@router.post("/model-workers/result")
async def complete_model_task(request: Request):
    _authorize(request, "MODEL_WORKER_TOKEN")
    body = await _read_json(request)
    if set(body) - {"task_id", "worker_id", "lease_token", "result", "request_hash", "audit"}:
        _error(422, "Worker result fields are invalid.")
    task_id, worker, lease = _lease_fields(body)
    request_hash = body.get("request_hash")
    if not isinstance(request_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", request_hash):
        _error(422, "Worker result input hash is invalid.")
    store = get_model_task_store()
    task = await _broker(store.get, task_id)
    if task["status"] not in {"running", "succeeded"}:
        _error(409, "Worker task lease is stale or no longer available.")
    if not secrets.compare_digest(task["request_hash"], request_hash):
        _error(409, "Worker result input hash does not match its task.")
    result = _validated_result(task, body.get("result"))
    audit = body.get("audit", {})
    if not isinstance(audit, dict) or set(audit) - {"model_digest", "response_model_inferred"}:
        _error(422, "Worker result audit is invalid.")
    if "model_digest" in audit and (not isinstance(audit["model_digest"], str) or not DIGEST.fullmatch(audit["model_digest"])):
        _error(422, "Worker result model digest is invalid.")
    if "response_model_inferred" in audit and type(audit["response_model_inferred"]) is not bool:
        _error(422, "Worker inferred model flag is invalid.")
    audit = {**task.get("request_audit", {}), **audit}
    completed = await _broker(store.complete, task_id, worker, lease, result, audit=audit, request_hash=request_hash)
    return {"status": completed["status"]}


@router.post("/model-workers/failure")
async def fail_model_task(request: Request):
    _authorize(request, "MODEL_WORKER_TOKEN")
    body = await _read_json(request)
    if set(body) - {"task_id", "worker_id", "lease_token", "category", "retryable"}:
        _error(422, "Worker failure fields are invalid.")
    category, retryable = body.get("category"), body.get("retryable", False)
    if not isinstance(category, str) or category not in FAILURE_CATEGORIES or type(retryable) is not bool:
        _error(422, "Worker failure category is invalid.")
    task = await _broker(get_model_task_store().fail, *_lease_fields(body), category, retryable=retryable)
    return {"status": task["status"]}


@router.get("/model-workers/status")
async def model_worker_status(request: Request):
    _authorize(request, "MODEL_GATEWAY_TOKEN")
    return await _broker(get_model_task_store().status)
