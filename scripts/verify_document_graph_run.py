"""Read-only structural verification of an existing document graph and its evidence."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import urlopen
import uuid


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_graph(graph, batches):
    """Check every candidate, endpoint and quoted Unicode slice; no model calls."""
    units = {}
    for batch in batches:
        for unit in batch["units"]:
            key = unit["unit_id"]
            require(key not in units or units[key] == unit, "Conflicting canonical unit: " + key)
            require(sha(unit["text"]) == unit["text_sha256"], "Unit text hash mismatch: " + key)
            units[key] = unit
    indexed = {}
    for field, key in (("sources", "source_id"), ("entities", "entity_id"), ("assertions", "assertion_id"), ("evidence", "evidence_id")):
        indexed[field] = {item[key]: item for item in graph[field]}
        require(len(indexed[field]) == len(graph[field]), "Duplicate identifiers in " + field)
    evidence, entities, sources = indexed["evidence"], indexed["entities"], indexed["sources"]
    for item in evidence.values():
        key = item["evidence_id"]
        unit = units.get(item["unit_id"])
        require(unit is not None, "Unregistered evidence unit: " + key)
        require(item["source_id"] in sources and all(item[k] == unit[k] for k in ("source_id", "source_version_id")), "Evidence ownership mismatch: " + key)
        require(sources[item["source_id"]]["source_version_id"] == item["source_version_id"], "Evidence source version mismatch: " + key)
        require(sha(item["text"]) == item["text_sha256"] and sha(unit["text"]) == item["unit_text_sha256"], "Evidence hash mismatch: " + key)
        locator = item["character_locator"]
        require(locator.get("coordinate_system") == "unit_text" and locator.get("offset_unit") == "unicode_code_points", "Unsupported quote coordinates: " + key)
        require(item["text"].strip() and locator["spans"], "Empty quotation: " + key)
        for span in locator["spans"]:
            start, end = span["start"], span["end"]
            require(type(start) is int and type(end) is int and 0 <= start < end <= len(unit["text"]), "Quote offsets out of bounds: " + key)
            require(unit["text"][start:end] == item["text"], "Quote differs from Unicode source slice: " + key)
    for field in ("entities", "assertions"):
        for item in graph[field]:
            require(item["source_id"] in sources and sources[item["source_id"]]["source_version_id"] == item["source_version_id"], "Candidate source ownership mismatch")
            require(item["evidence_ids"] and all(key in evidence and all(evidence[key][k] == item[k] for k in ("source_id", "source_version_id")) for key in item["evidence_ids"]), "Candidate evidence ownership mismatch")
            if field == "assertions":
                require(("object_id" in item) != ("value" in item), "Relationship requires one entity object or literal value")
                claim_units = {evidence[key]["unit_id"] for key in item["evidence_ids"]}
                for key in ("subject_id", "object_id"):
                    if key not in item:
                        continue
                    endpoint = entities.get(item[key])
                    require(endpoint is not None and all(endpoint[k] == item[k] for k in ("source_id", "source_version_id")), "Relationship endpoint ownership mismatch")
                    require(claim_units & {evidence[key]["unit_id"] for key in endpoint["evidence_ids"]}, "Relationship lacks endpoint evidence-unit overlap")
    per_source = []
    for source in sources.values():
        coverage = source.get("transcription_coverage", {})
        pdf = source.get("source_kind") == "file_pdf"
        per_source.append({"source_id": source["source_id"], "title": source.get("title"),
            "entities": sum(item["source_id"] == source["source_id"] for item in entities.values()),
            "relationships": sum(item["source_id"] == source["source_id"] for item in graph["assertions"]),
            "quotations": sum(item["source_id"] == source["source_id"] for item in evidence.values()),
            "readable_pages": coverage.get("readable_units") if pdf else None,
            "total_pages": coverage.get("total_units") if pdf else None,
            "transcription_coverage": coverage})
    return {"sources": len(sources), "entities": len(entities), "relationships": len(graph["assertions"]),
            "quotations": len(evidence), "exact_quote_offset_checks": len(evidence), "per_source": per_source,
            "readable_pages": sum(row["readable_pages"] or 0 for row in per_source),
            "total_pages": sum(row["total_pages"] or 0 for row in per_source), "integrity_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True, help="Backend API base including /api/v1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.api.rstrip("/")
    url = urlsplit(base)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        parser.error("Use an HTTP(S) API base without credentials, query or fragment")
    run_id = str(uuid.UUID(args.run_id))
    def get(path):
        with urlopen(base + path, timeout=60) as response:
            return json.load(response)
    prefix = "/research-runs/" + run_id
    run = get(prefix)
    names = set(run["artifacts"])
    graph_name = next((name for name in ("graph_input.json", "validated_preview.json") if name in names), None)
    require(graph_name is not None, "No graph or validated partial preview is available")
    selected = sorted({graph_name} | {name for name in names if name.endswith("_batches.json") or name.startswith("rejected_batches/") and name.endswith(".json")})
    with ThreadPoolExecutor(max_workers=4) as pool:
        artifacts = dict(zip(selected, pool.map(lambda name: get(prefix + "/artifacts/" + quote(name, safe="/")), selected)))
    batches = [batch for name, data in artifacts.items() if name.endswith("_batches.json") for batch in data]
    graph = artifacts[graph_name]
    report = verify_graph(graph, batches)
    batch_ids = {batch["batch_id"] for batch in batches}
    completed = {name.split("/")[-1][:-5] for name in names if name.startswith("processed/") and name.endswith(".json")} & batch_ids
    rejected = {name.split("/")[-1][:-5]: data for name, data in artifacts.items() if name.startswith("rejected_batches/")}
    active_rejected = set(rejected) & (batch_ids - completed)
    reasons = Counter(item["reason"] for key in active_rejected for item in rejected[key]["validation"]["rejections"])
    report.update(run_id=run_id, api=base, checked_at=datetime.now(timezone.utc).isoformat(),
        status=run["status"], artifact=graph_name, partial_extraction=bool(graph.get("partial_extraction") or graph_name == "validated_preview.json"),
        accepted_batch_count=len(completed), rejected_batch_count=len(active_rejected), pending_batch_count=len(batch_ids - completed),
        saved_rejected_batch_count=len(rejected), rejection_reason_histogram=dict(sorted(reasons.items())),
        scope="Structural candidate and quotation integrity only; accepted batches passed extraction validation, not scientific review. GET requests only.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"per_source", "api"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
