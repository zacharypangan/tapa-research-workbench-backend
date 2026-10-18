"""Small-corpus graph discovery with original quotations and explicit review status.

This is a bounded scan of graph checkpoints, independent of passage ranking.
It never promotes candidate or inclusion review to scientific acceptance.
"""
from __future__ import annotations

import re
import logging
import unicodedata

from .store import read_json

LOGGER = logging.getLogger(__name__)


def _normal(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def search_graph_entities(query: str, material_id: str | None = None, limit: int = 10, store=None) -> list[dict]:
    if not isinstance(query, str) or not query.strip() or not 1 <= limit <= 100:
        return []
    if store is None:
        from .jobs import shared_store
        store = shared_store()
    needle = _normal(query)
    tokens = re.findall(r"[\w]+", needle)
    found, seen = [], set()
    # The page limit is explicit; no query can inspect unbounded historical runs.
    runs = store.list_runs(100, material_id=material_id)["items"]
    for run in runs:
        path = store.path(run["run_id"])
        released = run.get("status") == "released" and run.get("release_path")
        artifact = path / run["release_path"] / "graph.json" if released else path / "graph_input.json"
        try:
            graph = read_json(artifact, {})
            review_rows = read_json(path / "decisions.json", [])
        except (ValueError, OSError):
            LOGGER.warning("Skipping unreadable graph-search checkpoint for run=%s", run["run_id"])
            continue
        if not graph:
            continue
        sources = {source["source_id"]: source for source in graph.get("sources", [])}
        eligible_sources = set(sources)
        if material_id is not None:
            eligible_sources = {identifier for identifier, source in sources.items()
                                if str(source.get("source_locator", "")).startswith(f"workbench:material/{material_id}/file/")}
            seed_ids = {entity["entity_id"] for entity in graph.get("entities", []) if entity["source_id"] in eligible_sources}
            eligible_sources |= {identifier for identifier, source in sources.items()
                                 if source.get("origin") == "web" and seed_ids.intersection(source.get("seed_ids", []))}
        evidence = {item["evidence_id"]: item for item in graph.get("evidence", [])}
        decisions = {item["candidate_id"]: item["decision"] for item in review_rows}
        for entity in graph.get("entities", []):
            if entity["source_id"] not in eligible_sources:
                continue
            key = (entity["entity_id"], sources.get(entity["source_id"], {}).get("source_locator"))
            if key in seen or decisions.get(entity["entity_id"]) == "reject":
                continue
            labels = [str(entity.get(name, "")) for name in ("label_original", "label_en")] + list(entity.get("aliases", []))
            normalized = [_normal(label) for label in labels]
            score = max((3.0 if label == needle else 2.0 if needle in label else
                         sum(token in label for token in tokens) / max(1, len(tokens))) for label in normalized)
            if score <= 0:
                continue
            source = sources.get(entity["source_id"], {})
            quotations = [evidence[identifier] for identifier in entity.get("evidence_ids", []) if identifier in evidence]
            if not quotations:
                continue
            seen.add(key)
            related = [claim for claim in graph.get("assertions", [])
                       if entity["entity_id"] in {claim.get("subject_id"), claim.get("object_id")}
                       and decisions.get(claim.get("assertion_id")) != "reject"]
            found.append({"run_id": run["run_id"], "entity_id": entity["entity_id"],
                          "label": entity.get("label_original"), "label_en": entity.get("label_en"),
                          "entity_type": entity["entity_type"], "aliases": entity.get("aliases", []),
                          "source_id": entity["source_id"], "source_locator": source.get("source_locator"),
                          "source_title": source.get("title"), "evidence": quotations[:3],
                          "relationships": related[:10], "score": score,
                          "publication_status": "released" if released else "draft",
                          "inclusion_status": entity.get("inclusion_review_status", decisions.get(entity["entity_id"], "pending")),
                          "scientific_status": "needs_specialist_review"})
    found.sort(key=lambda item: (-item["score"], item["label"] or "", item["run_id"]))
    return found[:limit]
