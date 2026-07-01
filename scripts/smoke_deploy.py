#!/usr/bin/env python3
"""Smoke check a deployed Research Workbench backend."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


CHECKS = [
    ("health", "/health"),
    ("repository statuses", "/api/v1/repository/statuses"),
    ("repository materials", "/api/v1/repository/materials"),
    ("progress portal", "/api/v1/progress/status"),
]


def normalize_base_url(value: str) -> str:
    base_url = value.rstrip("/")
    if base_url.endswith("/api/v1"):
        base_url = base_url[: -len("/api/v1")]
    return base_url


def fetch_json(url: str, timeout: float, token: str = "") -> tuple[int, object]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
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
    parser.add_argument(
        "--token",
        default=os.getenv("SMOKE_AUTH_TOKEN", ""),
        help="Clerk session token for protected checks; defaults to SMOKE_AUTH_TOKEN.",
    )
    args = parser.parse_args()

    base_url = normalize_base_url(args.base_url)
    failed = False

    for label, path in CHECKS:
        url = f"{base_url}{path}"
        try:
            token = "" if path == "/health" else args.token
            status, payload = fetch_json(url, args.timeout, token)
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
