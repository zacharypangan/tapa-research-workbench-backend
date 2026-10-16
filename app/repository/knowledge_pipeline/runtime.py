"""Run the collector in an existing operator-approved Python environment.

The API receives no executable-path option. A worker can be configured with
``collector_for_runtime(Path(...))`` by its operator, or use local discovery.
This module installs nothing and does not require scraping libraries on import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from urllib.parse import urlsplit
from uuid import uuid4


BACKEND_ROOT = Path(__file__).resolve().parents[3]
APPROVED_LOCAL_RUNTIME = BACKEND_ROOT.parent / "tmp" / "tapa-web-pilot-runtime-20261009" / "bin" / "python"
MAX_FRONTIER_PAGES = 25
MAX_COLLECTION_SECONDS = 1800
MAX_REQUEST_BYTES = 2 * 1024 * 1024


class CollectorRuntimeError(RuntimeError):
    """An existing collector interpreter is missing or failed to complete."""


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(BACKEND_ROOT) + (os.pathsep + previous if previous else "")
    return environment


def _python_path(python: Path) -> Path:
    # Keep venv symlink paths: resolving one would select the base interpreter.
    selected = Path(python).expanduser().absolute()
    if not selected.is_file() or not os.access(selected, os.X_OK):
        raise CollectorRuntimeError("Collector Python is not an existing executable; no runtime was installed")
    return selected


def _available_runtime(python: Path) -> bool:
    try:
        selected = _python_path(python)
        checked = subprocess.run([str(selected), "-c", "import bs4, httpx"],
                                 cwd=BACKEND_ROOT, env=_worker_environment(),
                                 capture_output=True, text=True, timeout=10, check=False)
        return checked.returncode == 0
    except (CollectorRuntimeError, OSError, subprocess.TimeoutExpired):
        return False


def existing_collector_runtime() -> Path:
    """Find current or previously approved local Python; never download one."""
    candidates = [Path(sys.executable), APPROVED_LOCAL_RUNTIME]
    for candidate in dict.fromkeys(candidates):
        if _available_runtime(candidate):
            return candidate.absolute()
    raise CollectorRuntimeError(
        "No installed collector runtime with BeautifulSoup and httpx is available; "
        "configure an existing approved interpreter with the operator --collector-python option"
    )


def _collection_timeout(frontier: list[dict], options: dict) -> int:
    from .adapters import _frontier

    entries = _frontier(frontier)
    if not isinstance(options, dict):
        raise ValueError("Collector options must be an object")
    if set(options) & {"python", "collector_python", "browser_binary", "driver_path"}:
        raise ValueError("Executable paths are operator configuration, not collection options")
    pages = options.get("max_pages", len(entries))
    if type(pages) is not int or not len(entries) <= pages <= MAX_FRONTIER_PAGES:
        raise ValueError("Runtime collection requires max_pages covering 1–25 explicit frontier URLs")
    timeout, delay = options.get("timeout", 20.0), options.get("delay", 1.0)
    for value, low, high, name in ((timeout, 1, 120, "timeout"), (delay, 0, 60, "delay")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(name + " must be within the permitted finite range")
    hosts = len({urlsplit(row["url"]).hostname for row in entries})
    # Two page fetch/render allowances plus one robots allowance per host.
    # A hard ceiling also bounds redirect/retry overhead in the existing adapter.
    return min(MAX_COLLECTION_SECONDS, math.ceil(30 + pages * (2 * timeout + delay) + hosts * timeout))


def _write_json(path: Path, payload: dict):
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
    path.chmod(0o400)


def collector_for_runtime(python: Path):
    """Return a synchronous ``(frontier, output_dir, options)`` collector."""
    selected = _python_path(python)

    def collect(frontier: list[dict], output_dir: Path, options: dict) -> dict:
        bounded_timeout = _collection_timeout(frontier, options)
        destination = Path(output_dir).absolute()
        if destination.exists() or destination.is_symlink():
            raise ValueError("Choose a fresh collection output directory")
        parent_directory = destination.parent
        parent_directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        invocation_id = uuid4().hex
        request_path = parent_directory / ("runtime-request-" + invocation_id + ".json")
        result_path = parent_directory / ("runtime-result-" + invocation_id + ".json")
        _write_json(request_path, {"frontier": frontier, "options": options, "output_dir": str(destination)})
        try:
            completed = subprocess.run(
                [str(selected), "-m", "app.repository.knowledge_pipeline.runtime", "collect",
                 "--request", str(request_path), "--result", str(result_path)],
                cwd=BACKEND_ROOT, env=_worker_environment(), capture_output=True,
                text=True, timeout=bounded_timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CollectorRuntimeError(
                f"Collector exceeded its bounded {bounded_timeout}-second runtime; preserved request and partial capture"
            ) from exc
        except OSError as exc:
            raise CollectorRuntimeError("Approved collector interpreter could not be started") from exc
        # stdout/stderr are deliberately never propagated or logged.
        if not result_path.is_file():
            raise CollectorRuntimeError("Collector worker exited without a structured result; inspect the preserved request")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CollectorRuntimeError("Collector worker returned an unreadable structured result") from exc
        if not isinstance(payload, dict):
            raise CollectorRuntimeError("Collector worker returned an invalid structured result")
        if payload.get("ok") is not True:
            error = payload.get("error", {})
            kind = error.get("type") if isinstance(error, dict) else None
            if kind == "ImportError":
                raise ImportError("Approved collector runtime is missing a requested extraction dependency; no dependency was installed")
            if kind == "ValueError":
                raise ValueError("Collector rejected its source, options or saved-snapshot validation; inspect the structured result")
            raise CollectorRuntimeError("Collector worker failed; inspect its structured error result")
        result = payload.get("result")
        if completed.returncode != 0 or not isinstance(result, dict) or any(not isinstance(result.get(key), list) for key in ("sources", "units", "attempts")):
            raise CollectorRuntimeError("Collector worker returned inconsistent completion metadata")
        result["runtime_execution"] = {
            "python": str(selected), "python_version": payload.get("python_version"),
            "request_path": str(request_path), "result_path": str(result_path),
            "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "timeout_seconds": bounded_timeout,
        }
        return result

    return collect


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("collect", help="Collect a seed-bound JSON frontier")
    command.add_argument("--request", required=True, type=Path)
    command.add_argument("--result", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.result.exists() or args.result.is_symlink():
        return 2
    try:
        if args.request.stat().st_size > MAX_REQUEST_BYTES:
            raise ValueError("Collection request exceeds the 2 MiB limit")
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(request, dict) or set(request) != {"frontier", "options", "output_dir"}:
            raise ValueError("Malformed collection request")
        _collection_timeout(request["frontier"], request["options"])
        if not isinstance(request["output_dir"], str) or not request["output_dir"]:
            raise ValueError("Collection output directory is missing")
        from .adapters import collect_frontier

        result = collect_frontier(request["frontier"], Path(request["output_dir"]), request["options"])
        payload = {"ok": True, "result": result, "python_version": platform.python_version()}
        exit_code = 0
    except Exception as exc:
        # Only controlled error classes are disclosed; parser/backend exception
        # strings can contain source URLs or runtime details and are not copied.
        kind = "ImportError" if isinstance(exc, ImportError) else "ValueError" if isinstance(exc, ValueError) else "CollectorRuntimeError"
        payload = {"ok": False, "error": {"type": kind}, "python_version": platform.python_version()}
        exit_code = 2
    _write_json(args.result, payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
