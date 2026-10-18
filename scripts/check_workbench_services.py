"""Repeatable, content-minimizing checks for an existing Workbench deployment.

Uses GET only. --probe-ai invokes the existing endpoint's small provider health
requests; it does not index, extract, resume, review or publish documents.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def check(base, path, timeout):
    start = time.monotonic()
    result = {"path": path}
    try:
        request = Request(base + path, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            result["status"] = response.status
            data = json.load(response)
        if path.endswith("/ai/status"):
            result["result"] = {key: value for key, value in data.items() if "url" not in key}
            result["ready"] = bool(data.get("chat_configured") and data.get("embedding_configured"))
        elif path.endswith("/capabilities") or path == "/health":
            result["result"] = data
        elif path == "/openapi.json":
            result["route_count"] = len(data.get("paths", {}))
        elif "/research-runs?" in path:
            result["total"] = data.get("total")
            result["runs"] = [{key: run.get(key) for key in ("run_id", "status", "stage")}
                              for run in data.get("items", [])]
        elif isinstance(data, dict):
            result["response_keys"] = list(data)
            result["total"] = data.get("total")
            for key in ("results", "materials"):
                if isinstance(data.get(key), list):
                    result["returned_" + key] = len(data[key])
        else:
            result["response_type"] = type(data).__name__
    except HTTPError as exc:
        result.update(status=exc.code, error="HTTP request failed")
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        result["error"] = type(exc).__name__
    result["seconds"] = round(time.monotonic() - start, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Backend origin, without /api/v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-ai", action="store_true", help="Also perform provider health requests")
    parser.add_argument("--timeout", type=int, default=60, choices=range(1, 121), metavar="1..120")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        parser.error("Use an http(s) backend origin without credentials, path, query or fragment")
    paths = ["/health", "/openapi.json", "/api/v1/repository/source-types",
             "/api/v1/repository/statuses", "/api/v1/repository/materials?limit=1",
             "/api/v1/repository/search?q=tapa&limit=5",
             "/api/v1/research-runs/capabilities", "/api/v1/research-runs?limit=20"]
    if args.probe_ai:
        paths.append("/api/v1/repository/ai/status")
    with ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(lambda path: check(base, path, args.timeout), paths))
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "base": base,
              "scope": "Reachability, inventory and optional provider readiness; not an end-to-end content-quality test",
              "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for result in checks:
        readiness = " DEGRADED" if result.get("ready") is False else ""
        print(f"{result.get('status', 'ERROR')} {result['path']} ({result['seconds']}s){readiness}")
    return 0 if all(row.get("status") == 200 and "error" not in row and row.get("ready") is not False for row in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
