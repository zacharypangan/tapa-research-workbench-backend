#!/usr/bin/env python3
"""Queue bounded, source-attributed saved graph drafts; resume explicitly later.

Dry-run by default. Uses registered extraction snapshots for any original format.
Reports contain identifiers/counts only, never document passages or credentials.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


PRIORITY = ["7e27cc82-43d4-484e-a691-d9f39a14eb3a", "13ba2f2b-9b57-4ae1-acb2-bbe5b6865fc3",
            "fdf71de6-6ed6-4cb9-81e7-c0dc95d02fbe", "e27c22d5-6ae1-4284-aa89-7093a2b22f7b"]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def request(base, path, body=None, key=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Idempotency-Key"] = key
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urlopen(Request(base + path, data=data, headers=headers), timeout=60) as response:
            return json.load(response)
    except HTTPError as exc:
        payload = json.loads(exc.read())
        raise RuntimeError(f"HTTP {exc.code}: {payload.get('detail', 'Request failed')}") from exc


def graph_counts(base, run):
    names = set(run.get("artifacts", []))
    artifact = next((name for name in [f"{run.get('release_path')}/graph.json", "graph_input.json", "validated_preview.json"] if name in names), None)
    if artifact:
        data = request(base, f"/research-runs/{run['run_id']}/artifacts/{artifact}")
        return {"entities": len(data.get("entities", [])), "relationships": len(data.get("assertions", [])),
                "quotations": len(data.get("evidence", [])), "partial": data.get("partial_extraction", False), "artifact": artifact}
    checkpoints = [name for name in names if name.startswith("processed/") and name.endswith(".json")]
    complete = {name.split("/")[-1] for name in checkpoints}
    checkpoints += [name for name in names if name.startswith("validated_candidates/") and name.endswith(".json") and name.split("/")[-1] not in complete]
    with ThreadPoolExecutor(max_workers=3) as pool:
        data = list(pool.map(lambda name: request(base, f"/research-runs/{run['run_id']}/artifacts/{name}"), checkpoints[:100]))
    return {"entities": len({row["entity_id"] for part in data for row in part.get("entities", [])}),
            "relationships": len({row["assertion_id"] for part in data for row in part.get("assertions", [])}),
            "quotations": len({row["evidence_id"] for part in data for row in part.get("evidence", [])}),
            "partial": bool(data), "artifact": "live_batch_checkpoints" if data else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--material-id", action="append")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-auto-batches", type=int, default=2)
    parser.add_argument("--batch-chars", type=int, default=4000)
    parser.add_argument("--key-prefix", default="demo-precompute-2026-10-19")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()
    base = args.api.rstrip("/")
    url = urlsplit(base)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        parser.error("Use an HTTP(S) API base without credentials, query or fragment")
    if not 1 <= args.max_auto_batches <= 100 or not 1000 <= args.batch_chars <= 24000:
        parser.error("Use 1–100 automatic batches and 1000–24000 characters per batch")
    capability = request(base, "/research-runs/capabilities")
    if not capability.get("registered_extraction", {}).get("available"):
        parser.error("Backend does not yet expose registered-extraction graph creation; deploy this implementation first")
    ontology = request(base, "/research-runs/ontologies/general")
    config = {"ontology": ontology, "research_topic": "documented research evidence",
              "processing": {"mode": "ollama" if capability["extraction"]["configured"] else "manual",
                             "batch_chars": args.batch_chars, "max_auto_batches": args.max_auto_batches},
              "augmentation": {"enabled": False}}
    materials = request(base, "/repository/materials?limit=500")["materials"]
    if args.material_id:
        selected = set(args.material_id)
        materials = [material for material in materials if material["id"] in selected]
        if {material["id"] for material in materials} != selected:
            parser.error("One or more selected materials were not found")
    materials.sort(key=lambda material: (PRIORITY.index(material["id"]) if material["id"] in PRIORITY else len(PRIORITY),
                                         material.get("segment_count", 0), material["id"]))
    report = {"api": base, "created_at": datetime.now(timezone.utc).isoformat(), "executed": args.execute,
              "method": "Bounded automatic drafts from immutable registered extraction snapshots; all original passages retained",
              "config": {"batch_chars": args.batch_chars, "max_auto_batches": args.max_auto_batches}, "documents": []}
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)
    pending = []
    for material in materials:
        row = {"material_id": material["id"], "title": material["title"], "segment_count": material.get("segment_count", 0)}
        report["documents"].append(row)
        try:
            if not material.get("segment_count"):
                row["action"] = "needs_extraction"
                continue
            saved = request(base, "/research-runs?" + urlencode({"material_id": material["id"], "limit": 100}))["items"]
            own = next((item for item in saved if item.get("has_saved_candidates")), None)
            own = own or next((item for item in saved if item.get("pipeline_version") == "research-kg/1.1" and
                (item.get("title", "").startswith("Demo saved draft — ") or item.get("max_auto_batches") == args.max_auto_batches)), None)
            if own:
                run = request(base, "/research-runs/" + own["run_id"])
                row.update({"run_id": run["run_id"], "status": run["status"], "action": "reused_saved_graph", **graph_counts(base, run)})
                if run["status"] in {"created", "queued", "running"}:
                    pending.append(row)
                continue
            detail = request(base, "/repository/materials/" + material["id"])
            refs = [file["id"] for file in detail["files"]]
            body = {"material_id": material["id"], "file_ids": refs, "title": ("Demo saved draft — " + material["title"])[:200], "config": config}
            # The endpoint snapshots the full extraction, never only the first N pages.
            key = args.key_prefix + ":" + digest([body, capability["extraction"].get("processing_version")])
            row["action"] = "queue_registered_extraction" if args.execute else "would_queue_registered_extraction"
            if args.execute:
                run = request(base, "/research-runs/from-extraction", body, key)
                row.update({"run_id": run["run_id"], "status": run["status"]})
                pending.append(row)
        except Exception as exc:
            row.update({"action": "error", "error": str(exc)[:500]})
        finally:
            save()
            print(json.dumps(row, ensure_ascii=False), flush=True)
    deadline = time.monotonic() + max(0, args.timeout_seconds)
    while pending and time.monotonic() < deadline:
        for row in pending[:]:
            try:
                run = request(base, "/research-runs/" + row["run_id"])
                row.update({"status": run["status"], "automatic_batch_budget_reached": run.get("automatic_batch_budget_reached", False),
                            "missing_batch_count": len(run.get("missing_batch_ids", [])),
                            "rejected_batch_count": len(run.get("rejected_batch_ids", [])), **graph_counts(base, run)})
                if run["status"] not in {"created", "queued", "running"}:
                    pending.remove(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
            except Exception as exc:
                row["poll_error"] = str(exc)[:500]
            save()
        if pending:
            time.sleep(min(max(1, args.poll_seconds), max(0, deadline - time.monotonic())))
    report["pending_run_ids"] = [row["run_id"] for row in pending]
    save()
    print(json.dumps({"saved": str(args.output), "documents": len(report["documents"]), "pending": len(pending)}), flush=True)


if __name__ == "__main__":
    main()
