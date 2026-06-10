#!/usr/bin/env python3
"""Smoke check a deployed Research Workbench backend."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


CHECKS = [
    ("health", "/health"),
    ("repository statuses", "/api/v1/repository/statuses"),
    ("repository materials", "/api/v1/repository/materials"),
]


def normalize_base_url(value: str) -> str:
    base_url = value.rstrip("/")
    if base_url.endswith("/api/v1"):
        base_url = base_url[: -len("/api/v1")]
    return base_url


def fetch_json(url: str, timeout: float) -> tuple[int, object]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = raw
        return response.status, payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check health and core repository routes for a backend deployment.",
    )
    parser.add_argument(
        "base_url",
        nargs="?",
        default="http://127.0.0.1:8000",
        help="Backend root URL, for example https://example.up.railway.app",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    base_url = normalize_base_url(args.base_url)
    failed = False

    for label, path in CHECKS:
        url = f"{base_url}{path}"
        try:
            status, payload = fetch_json(url, args.timeout)
            ok = 200 <= status < 300
        except (urllib.error.URLError, TimeoutError) as exc:
            ok = False
            status = "error"
            payload = str(exc)

        marker = "OK" if ok else "FAIL"
        print(f"[{marker}] {label}: {url} ({status})")
        if not ok:
            failed = True
            print(f"      {payload}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
