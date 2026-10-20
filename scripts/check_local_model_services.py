"""Bounded Ollama service conformance checks; never installs or downloads models.

Uses synthetic evidence only. Passing is a smoke test, not a scientific quality
evaluation or a claim that the production Workbench is connected to this server.
"""

import argparse
import base64
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import struct
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zlib


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OllamaClient:
    def __init__(self, base, key=None, timeout=90):
        self.base = base.rstrip("/")
        if not self.base.endswith("/api"):
            self.base += "/api"
        self.key, self.timeout = key, timeout
        self.opener = build_opener(NoRedirect())

    def __call__(self, path, payload=None):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = Request(self.base + "/" + path, data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
        with self.opener.open(request, timeout=self.timeout) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("Response exceeds smoke-test limit")
        return json.loads(raw)


def red_square():
    """A synthetic 128px red image, with no research document contents."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    raw = (b"\0" + bytes((255, 0, 0)) * 128) * 128
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 128, 128, 8, 2, 0, 0, 0))
                            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")).decode()


def message(data):
    value = data.get("message", {}).get("content")
    if not isinstance(value, str) or not value.strip() or data.get("done") is False:
        raise ValueError("Missing or incomplete model answer")
    return value.strip()


def run_checks(client, chat_model, embedding_model, vision_model):
    results = {}
    def check(name, model, operation, **extra):
        if not model:
            results[name] = {"state": "not_configured", "ready": False, **extra}
            return
        start = time.monotonic()
        try:
            detail = operation(model)
            results[name] = {"state": "passed", "ready": True, "model": model, **extra, **(detail or {})}
        except Exception as exc:
            results[name] = {"state": "failed", "ready": False, "model": model,
                             "error_category": "http_error" if isinstance(exc, HTTPError) else type(exc).__name__,
                             **({"http_status": exc.code} if isinstance(exc, HTTPError) else {}), **extra}
        results[name]["seconds"] = round(time.monotonic() - start, 3)

    def chat(model, prompt, **kwargs):
        return message(client("chat", {"model": model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "think": False, "keep_alive": 0, "options": {"temperature": 0, "num_predict": 256, "num_ctx": 4096}, **kwargs}))

    def generation(model):
        answer = chat(model, "Return exactly this text without punctuation or explanation: workbench-ready")
        if answer.lower().strip(' .\"\n') != "workbench-ready":
            raise ValueError("Generation fixture failed")

    def extraction(model):
        source = "Researchers used a stone beater to make barkcloth."
        schema = {"type": "object", "properties": {key: {"type": "string"} for key in ("tool", "product", "quote")},
                  "required": ["tool", "product", "quote"], "additionalProperties": False}
        data = json.loads(chat(model, "Extract the tool, product and one exact supporting quote from this source. "
            "Use only source evidence. Return JSON matching this schema: " + json.dumps(schema) + "\nSource: " + source, format=schema))
        if (not isinstance(data, dict) or set(data) != {"tool", "product", "quote"}
                or data["tool"] != "stone beater" or data["product"] != "barkcloth"
                or not isinstance(data["quote"], str) or not data["quote"] or data["quote"] not in source
                or "stone beater" not in data["quote"] or "barkcloth" not in data["quote"]):
            raise ValueError("Structured evidence fixture failed")
        return {"schema_and_exact_quote_fixture": True, "full_workbench_validation": False}

    def translation(model):
        for source in ("这种石拍用于制作树皮布。", "この石製のたたき具は樹皮布を作るために使われます。"):
            answer = chat(model, "Translate this sentence to English. Return only the translation:\n" + source).lower()
            if "stone" not in answer or not any(term in answer for term in ("barkcloth", "bark cloth", "bark-cloth")):
                raise ValueError("Translation terminology fixture failed")
        return {"languages": ["zh", "ja"], "needs_human_review": True, "test": "terminology smoke check"}

    def embeddings(model):
        vectors = client("embed", {"model": model, "input": ["stone beater", "barkcloth production"], "truncate": False, "keep_alive": 0}).get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != 2:
            raise ValueError("Missing embedding batch")
        for vector in vectors:
            if (not isinstance(vector, list) or not vector or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector)
                    or sum(x * x for x in vector) == 0):
                raise ValueError("Invalid embedding")
        if len(vectors[0]) != len(vectors[1]):
            raise ValueError("Embedding dimensions differ")
        return {"dimensions": len(vectors[0]), "retrieval_quality_evaluated": False}

    def vision(model):
        payload = {"model": model, "stream": False, "think": False, "keep_alive": 0,
                   "options": {"temperature": 0, "num_predict": 32, "num_ctx": 4096},
                   "messages": [{"role": "user", "content": "What is the dominant color? Return just the color name in English.", "images": [red_square()]}]}
        if message(client("chat", payload)).lower().strip(' .\"\n') != "red":
            raise ValueError("Vision color fixture failed")
        return {"research_figure_quality_evaluated": False}

    check("generation", chat_model, generation)
    check("structured_extraction", chat_model, extraction)
    check("translation", chat_model, translation)
    check("embeddings", embedding_model, embeddings)
    check("vision", vision_model, vision)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--chat-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--vision-model", required=True)
    parser.add_argument("--api-key-env", help="Optional existing environment variable NAME; never pass a key as an argument")
    parser.add_argument("--timeout", type=int, default=90, choices=range(1, 121), metavar="1..120")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment
            or url.path.rstrip("/") not in {"", "/api"}):
        parser.error("Use an http(s) model origin, optionally /api, without credentials or query parameters")
    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not key:
        parser.error("The named key variable is not set")
    client = OllamaClient(args.base_url, key, args.timeout)
    identity = {}
    try:
        identity["runtime_version"] = client("version")["version"]
        models = client("tags").get("models", [])
        requested = {args.chat_model, args.embedding_model, args.vision_model}
        identity["models"] = [{key: model.get(key) for key in ("name", "digest", "size")} for model in models
                              if model.get("name") in requested or model.get("model") in requested]
    except Exception as exc:
        identity["identity_error"] = type(exc).__name__
    results = run_checks(client, args.chat_model, args.embedding_model, args.vision_model)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "scope": "Synthetic conformance smoke tests; not production integration or research-quality evaluation",
              "runtime_identity": identity, "services": results, "ready": all(item["ready"] for item in results.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, result in results.items():
        print(name + ": " + result["state"])
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
