"""Derived, source-preserving networks over saved research graph candidates.

This module deliberately lives outside the pinned extraction implementation.
It does not execute extraction, merge physical identities or publish assertions.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
import re
import uuid

from app.repository.knowledge_pipeline.store import digest, file_hash, now, read_json
from app.repository.processing_jobs import lock, save

VERSION = "graph-network/1"
FIELDS = {"sources": "source_id", "entities": "entity_id", "evidence": "evidence_id",
          "assertions": "assertion_id", "alignments": "alignment_id"}
LOCATOR = re.compile(r"^workbench:material/([^/]+)/file/([^/]+)$")
GLOBAL_IDS = {"material_id", "file_id", "segment_id", "origin_run_id", "original_id", "run_id", "extraction_run_id"}
LOCAL_IDS = {"source_id", "entity_id", "evidence_id", "assertion_id", "alignment_id", "candidate_id",
             "seed_id", "unit_id", "source_version_id", "representation_version_id", "evidence_version_id",
             "entity_version_id", "assertion_version_id", "alignment_version_id", "candidate_version", "locator_id", "span_id", "local_id"}
REVIEW_FIELDS = {"review_status", "inclusion_review_status", "inclusion_decision", "review_decision",
                 "review_actor_id", "reviewed_at", "candidate_version"}


class NetworkSnapshotUnavailable(OSError):
    """A stored network could not be read safely; do not expose its path."""


def _all_runs(store) -> list[dict]:
    rows, offset = [], 0
    while True:
        page = store.list_runs(limit=100, offset=offset)
        rows.extend(page["items"])
        offset += len(page["items"])
        if offset >= page["total"] or not page["items"]:
            return rows


def _registered_documents() -> list[dict]:
    from app.api import repository
    with repository.get_connection() as connection:
        documents = [dict(row) for row in connection.execute(
            "SELECT id AS material_id,title,authors,year,source_type,collection,source_url,language,region "
            "FROM materials ORDER BY created_at,id")]
        files = [dict(row) for row in connection.execute(
            "SELECT f.id AS file_id,f.material_id,f.original_filename,COUNT(es.id) AS extracted_segment_count "
            "FROM files f LEFT JOIN extracted_segments es ON es.file_id=f.id "
            "AND es.source_kind NOT GLOB 'historical_*' AND LENGTH(TRIM(es.content_text))>0 "
            "GROUP BY f.id ORDER BY f.material_id,f.id")]
    for document in documents:
        document["files"] = [{**row, "locator": f"workbench:material/{row['material_id']}/file/{row['file_id']}"}
                             for row in files if row["material_id"] == document["material_id"]]
        document["file_count"] = len(document["files"])
        document["extracted_segment_count"] = sum(row["extracted_segment_count"] for row in document["files"])
    return documents


def _assert_current(path: Path, inputs: dict) -> None:
    """Reject altered snapshots or deleted/replaced registered originals."""
    from app.api import repository
    with repository.get_connection() as connection:
        for document in inputs["documents"]:
            snapshot = (path / document["input_path"]).resolve()
            if not snapshot.is_relative_to(path.resolve()) or not snapshot.is_file() or file_hash(snapshot) != document["sha256"]:
                raise ValueError("Research run has an altered or unavailable input snapshot")
            match = LOCATOR.fullmatch(document.get("original_locator", ""))
            if match is None:
                continue
            row = connection.execute("SELECT stored_path FROM files WHERE material_id=? AND id=?", match.groups()).fetchone()
            if row is None:
                raise ValueError("Research run refers to a deleted registered document")
            original = Path(row["stored_path"]).resolve()
            if not original.is_relative_to(Path(repository.FILES_ROOT).resolve()) or not original.is_file():
                raise ValueError("Research run original is unavailable in managed storage")
            expected = document["sha256"]
            if document.get("representation_kind") == "registered_extraction":
                expected = read_json(snapshot, {}).get("parent_file", {}).get("sha256")
            if not expected or file_hash(original) != expected:
                raise ValueError("Research run refers to a replaced document; generate a current graph first")


def _validate_references(graph: dict) -> None:
    indexed = {}
    for field, identifier in FIELDS.items():
        records = graph[field]
        indexed[field] = {row[identifier]: row for row in records}
        if len(indexed[field]) != len(records):
            raise ValueError("Duplicate graph identity")
    for field in ("entities", "evidence", "assertions", "alignments"):
        for row in graph[field]:
            source_id = row.get("source_id")
            if field != "alignments" and source_id not in indexed["sources"]:
                raise ValueError("Saved graph contains an unresolved source reference")
            if source_id and source_id not in indexed["sources"]:
                raise ValueError("Saved graph contains an unresolved source reference")
            if source_id and row.get("source_version_id") not in {None, indexed["sources"][source_id].get("source_version_id")}:
                raise ValueError("Saved graph contains a stale source version reference")
            if field != "evidence" and any(identifier not in indexed["evidence"] for identifier in row.get("evidence_ids", [])):
                raise ValueError("Saved graph contains an unresolved evidence reference")
            if field == "entities" and (not row.get("evidence_ids") or any(indexed["evidence"][identifier]["source_id"] != source_id for identifier in row["evidence_ids"])):
                raise ValueError("Saved concept evidence does not belong to its source")
            scope = row.get("scope") or {}
            for key in ("place_id", "time_id", "scheme_id", "production_case_id", "case_id", "typology_scheme_id"):
                if key in scope and scope[key] not in indexed["entities"]:
                    raise ValueError("Saved graph contains an unresolved context reference")
            if field in {"assertions", "alignments"}:
                if row.get("subject_id") not in indexed["entities"] or (row.get("object_id") is not None and row["object_id"] not in indexed["entities"]):
                    raise ValueError("Saved graph contains an unresolved concept endpoint")


def _graph_files(graph: dict) -> set[tuple[str, str]]:
    used = {row["source_id"] for row in graph["entities"]}
    return {tuple(match.groups()) for row in graph["sources"] if row["source_id"] in used
            and (match := LOCATOR.fullmatch(row.get("source_locator", "")))}


def _decision_time(decision: dict) -> float:
    try:
        value = datetime.fromisoformat(str(decision.get("at", "")).replace("Z", "+00:00"))
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).timestamp()
    except (ValueError, TypeError, OverflowError):
        return float("-inf")


def load_saved_graph(store, run_id: str) -> tuple[dict, dict]:
    """Read every safe checkpoint, including candidates after the listing hint's cap."""
    path = store.path(run_id)
    inputs, state = read_json(path / "inputs.json"), read_json(path / "state.json", {})
    _assert_current(path, inputs)
    published = state.get("status") == "released"
    if published:
        if not state.get("release_path"):
            raise ValueError("Published graph is unavailable; draft checkpoints cannot replace its release")
        preferred = [path / state["release_path"] / "graph.json"]
    else:
        preferred = [path / "graph_input.json", path / "validated_preview.json"]
    artifacts = []
    graph = {field: [] for field in FIELDS}
    metadata = {}
    for artifact in preferred:
        if artifact.is_file() and artifact.resolve().is_relative_to(path.resolve()):
            candidate = read_json(artifact)
            if candidate.get("entities"):
                metadata = candidate
                for field in FIELDS:
                    graph[field] = copy.deepcopy(candidate.get(field, []))
                artifacts.append(artifact)
                break
    indexed = {field: {row[key]: row for row in graph[field]} for field, key in FIELDS.items()}
    if published and not metadata:
        raise ValueError("Published graph is unavailable; draft checkpoints cannot replace its release")
    for directory in (() if published else ("processed", "validated_candidates")):
        for artifact in sorted((path / directory).glob("*.json")):
            if directory == "validated_candidates" and (path / "processed" / artifact.name).is_file():
                continue
            output = read_json(artifact)
            if directory == "processed" and output.get("rejections"):
                continue
            artifacts.append(artifact)
            for field in ("entities", "assertions", "evidence"):
                key = FIELDS[field]
                for row in output.get(field, []):
                    previous = indexed[field].get(row[key])
                    if previous is not None and any(previous.get(name) != value for name, value in row.items() if name not in REVIEW_FIELDS):
                        raise ValueError("Conflicting graph checkpoint versions")
                    indexed[field].setdefault(row[key], copy.deepcopy(row))
    for source in ([] if published else read_json(path / "ingest.json", {}).get("sources", [])):
        indexed["sources"].setdefault(source["source_id"], source)
    # Augmented checkpoint source records are separate from primary ingest.
    for artifact in ([] if published else sorted(path.rglob("source.json"))):
        source = read_json(artifact, {})
        if source.get("source_id"):
            indexed["sources"].setdefault(source["source_id"], source)
    descriptors = {row["source_id"]: row for row in inputs["documents"]}
    for identifier, source in indexed["sources"].items():
        if identifier in descriptors and not published:
            source["source_locator"] = descriptors[identifier]["original_locator"]
    graph = {field: list(values.values()) for field, values in indexed.items()}
    if not graph["entities"]:
        raise ValueError("Research run has no saved nonempty concept graph")
    _validate_references(graph)
    used = {row["source_id"] for row in graph["entities"]}
    for source in graph["sources"]:
        if source["source_id"] not in used:
            continue
        if source.get("semantic_eligible") is False or source.get("eligible_for_graph") is False:
            raise ValueError("Saved graph has a source ineligible for semantic reuse")
        if source.get("retention_expires_at"):
            expiry = datetime.fromisoformat(source["retention_expires_at"].replace("Z", "+00:00"))
            if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
                raise ValueError("Saved graph source retention has expired; refresh or exclude that source")
    batches = {batch["batch_id"] for artifact in path.glob("*batches.json") for batch in read_json(artifact, [])}
    accepted = {artifact.stem for artifact in (path / "processed").glob("*.json")}
    partial = {artifact.stem for artifact in (path / "validated_candidates").glob("*.json")} - accepted
    graph["partial_extraction"] = bool(metadata.get("partial_extraction") or batches - accepted or state.get("status") not in {"released", "awaiting_review"})
    graph["pending_batch_count"] = len(batches - accepted)
    original_decisions = read_json(path / "decisions.json", metadata.get("decisions", []))
    candidates = {row[key]: row for field, key in FIELDS.items() if field in {"entities", "assertions", "alignments"} for row in graph[field]}
    latest = {}
    for decision in original_decisions:
        candidate = candidates.get(decision.get("candidate_id"))
        if candidate is None or (decision.get("candidate_version") and decision["candidate_version"] != candidate.get("candidate_version")):
            continue
        prior = latest.get(decision["candidate_id"])
        if prior is None or _decision_time(decision) >= _decision_time(prior):
            latest[decision["candidate_id"]] = decision
    graph["origin_decisions"] = list(latest.values())
    return graph, {"run_id": run_id, "title": inputs.get("title"), "status": state.get("status"),
                   "created_at": inputs["created_at"], "input_fingerprint": inputs["fingerprint"],
                   "documents": inputs["documents"], "artifacts": [{"name": artifact.relative_to(path).as_posix(), "sha256": file_hash(artifact)} for artifact in artifacts],
                   "batch_coverage": {"total": len(batches), "accepted": len(batches & accepted), "partially_validated": len(batches & partial), "pending": len(batches - accepted)},
                   "original_decisions": original_decisions}


def namespace_graph(graph: dict, run_id: str) -> dict:
    """Namespace reference values, never quotation strings, locators or global repository IDs."""
    prefix = f"RUN:{run_id}:"
    identifiers = {row[key] for field, key in FIELDS.items() for row in graph[field]}
    def transform(value, key=""):
        if isinstance(value, dict):
            if key in {"evidence_version_ids", "entity_version_ids"}:
                return {prefix + name: prefix + child if isinstance(child, str) else transform(child) for name, child in value.items()}
            return {(prefix + name if key in {"evidence_version_ids", "entity_version_ids"} else name): transform(child, name)
                    for name, child in value.items()}
        if isinstance(value, list):
            return [transform(child, key) for child in value]
        singular = key[:-1] if key.endswith("_ids") else key
        if isinstance(value, str) and key not in GLOBAL_IDS and (singular in LOCAL_IDS or ((key.endswith("_id") or key.endswith("_ids")) and value in identifiers)):
            return prefix + value
        return value
    result = transform(graph)
    decisions = {row["candidate_id"]: row for row in graph.get("origin_decisions", [])}
    for field, identity in FIELDS.items():
        for original, row in zip(graph[field], result[field]):
            row["origin_run_id"] = run_id
            row["original_id"] = original[identity]
            row["origin_provenance"] = {"run_id": run_id, "original_id": original[identity],
                "original_review": {key: original[key] for key in REVIEW_FIELDS if key in original}}
            if field in {"entities", "assertions", "alignments"}:
                original_status = str(original.get("review_status", original.get("inclusion_decision", ""))).lower()
                if original_status in {"reject", "rejected", "exclude", "excluded"}:
                    row["origin_review_decision"] = "reject"
                if original[identity] in decisions:
                    row["origin_review_decision"] = decisions[original[identity]]["decision"]
                for key in REVIEW_FIELDS:
                    if key != "candidate_version":
                        row.pop(key, None)
                row["review_status"] = "needs_specialist_review"
    return result


class NetworkStore:
    def __init__(self, root: Path, run_store):
        self.root, self.runs = Path(root), run_store
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, network_id: str) -> Path:
        if str(uuid.UUID(network_id)) != network_id:
            raise ValueError("Invalid graph network ID")
        path = self.root / (network_id + ".json")
        if not path.is_file():
            raise FileNotFoundError("Graph network not found")
        return path

    def get(self, network_id: str) -> dict:
        return self._read_snapshot(self.path(network_id))

    @staticmethod
    def _read_snapshot(path: Path) -> dict:
        try:
            snapshot = read_json(path)
            if not isinstance(snapshot, dict) or not isinstance(snapshot.get("network"), dict) or not isinstance(snapshot.get("graph"), dict):
                raise ValueError("Invalid snapshot structure")
            if snapshot["network"].get("network_id") != path.stem or not snapshot["network"].get("created_at"):
                raise ValueError("Invalid snapshot identity")
            return snapshot
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            raise NetworkSnapshotUnavailable("Saved graph network is temporarily unavailable") from exc

    def list(self, limit=20, offset=0) -> dict:
        rows, unavailable = [], 0
        for path in self.root.glob("*.json"):
            if path.is_symlink():
                continue
            try:
                rows.append(self._read_snapshot(path)["network"])
            except NetworkSnapshotUnavailable:
                unavailable += 1
        rows.sort(key=lambda row: (row["created_at"], row["network_id"]), reverse=True)
        return {"items": rows[offset:offset + limit], "total": len(rows), "limit": limit, "offset": offset, "unavailable_count": unavailable}

    def create(self, title: str, run_ids: list[str], scope="selected", infer_connections=True) -> dict:
        if scope not in {"selected", "library"} or not title.strip() or len(title) > 200:
            raise ValueError("Supply a title and selected or library scope")
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("Select distinct research run IDs")
        if scope == "selected" and not 1 <= len(run_ids) <= 200:
            raise ValueError("Select between 1 and 200 saved graphs")
        if scope == "library" and run_ids:
            raise ValueError("Library scope selects current saved graphs automatically; omit run_ids")
        documents = _registered_documents()
        loaded, issues = {}, []
        if scope == "library":
            candidates = {}
            for run in _all_runs(self.runs):
                try:
                    graph, origin = load_saved_graph(self.runs, run["run_id"])
                except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
                    issues.append({"run_id": run["run_id"], "reason": str(exc)})
                    continue
                for material_id, _ in sorted(_graph_files(graph)):
                    rank = (len(graph["entities"]), len(graph["assertions"]), not graph["partial_extraction"], origin["created_at"], run["run_id"])
                    current = candidates.get(material_id)
                    if current is None or rank > current[0]:
                        candidates[material_id] = (rank, run["run_id"], graph, origin)
            for material_id in sorted(candidates):
                _, run_id, graph, origin = candidates[material_id]
                loaded[run_id] = (graph, origin)
        else:
            for run_id in run_ids:
                loaded[run_id] = load_saved_graph(self.runs, run_id)
        if not loaded:
            raise ValueError("No current saved nonempty graphs are available; extract documents and prepare graphs first")
        combined = {field: [] for field in FIELDS}
        combined["origin_decisions"] = []
        origins, included_files = [], set()
        for run_id in sorted(loaded):
            graph, origin = loaded[run_id]
            namespaced = namespace_graph(graph, run_id)
            for field in FIELDS:
                combined[field].extend(namespaced[field])
            combined["origin_decisions"].extend(namespaced.get("origin_decisions", []))
            origins.append(origin)
            included_files.update(_graph_files(graph))
        if scope == "selected":
            selected_materials = {match[1] for _, origin in loaded.values() for descriptor in origin["documents"] if (match := LOCATOR.fullmatch(descriptor["original_locator"]))}
            documents = [row for row in documents if row["material_id"] in selected_materials]
        for document in documents:
            material_id = document["material_id"]
            for file in document["files"]:
                file["included_in_graph"] = (material_id, file["file_id"]) in included_files
            document["covered_file_count"] = sum(file["included_in_graph"] for file in document["files"])
            document["run_ids"] = sorted(run_id for run_id, (graph, _) in loaded.items() if any(mid == material_id for mid, _ in _graph_files(graph)))
            document["run_id"] = document["run_ids"][0] if document["run_ids"] else None
            document["saved_account_count"] = len(document["run_ids"])
            document["status"] = "included" if document["run_ids"] else "no_extracted_text" if not document["extracted_segment_count"] else "no_saved_graph"
            document["partial_extraction"] = document["covered_file_count"] < document["file_count"] or any(loaded[run_id][0]["partial_extraction"] for run_id in document["run_ids"])
            document["pending_batch_count"] = sum(loaded[run_id][0]["pending_batch_count"] for run_id in document["run_ids"])
            document["detail"] = (f"Saved partial graph; {document['pending_batch_count']} source passages remain uncompleted. "
                                  f"{document['covered_file_count']}/{document['file_count']} files contribute concepts."
                                  if document["run_ids"] and document["partial_extraction"] else
                                  "Saved graph candidates are included and still require review." if document["run_ids"] else
                                  "No current saved graph; this document remains in the database register.")
            if document["saved_account_count"] > 1:
                document["detail"] += f" {document['saved_account_count']} independent saved accounts remain separate because selected runs include multiple documents."
        coverage = {"material_count": len(documents), "materials_with_graph": sum(row["status"] == "included" for row in documents),
                    "materials_without_graph": sum(row["status"] != "included" for row in documents),
                    "file_count": sum(row["file_count"] for row in documents),
                    "files_with_extraction": sum(file["extracted_segment_count"] > 0 for row in documents for file in row["files"]),
                    "covered_file_count": sum(row["covered_file_count"] for row in documents), "materials": documents,
                    "excluded_runs": sorted(issues, key=lambda row: (row["run_id"], row["reason"])),
                    "selection_policy": "Select the most populated current saved run per material. A selected multi-document run can bring additional independent accounts; those accounts retain separate IDs and provenance.",
                    "interpretation": "A saved snapshot of current graph candidates and the document register. Missing files, pending extraction and unprocessed passages remain explicit; this is not a completed scientific database."}
        combined.update(partial_extraction=any(graph["partial_extraction"] for graph, _ in loaded.values()) or coverage["covered_file_count"] < coverage["file_count"] or bool(coverage["materials_without_graph"]),
                        pending_batch_count=sum(graph["pending_batch_count"] for graph, _ in loaded.values()), decisions=[], release_kind="derived_candidate_snapshot")
        combined["inferences"] = []
        inference_report = {"enabled": False, "added_connections": 0}
        if infer_connections:
            from app.repository.graph_inferences import infer_connections as infer
            inferred = infer(combined)
            combined["inferences"] = inferred["inferences"]
            inference_report = inferred["report"]
        counts = {field: len(combined[field]) for field in FIELDS}
        counts["inferences"] = len(combined["inferences"])
        counts["inferred_connections"] = len(combined["inferences"])
        fingerprint = digest({"version": VERSION, "title": title.strip(), "scope": scope,
                              "graph": combined, "coverage": coverage, "infer_connections": infer_connections})
        network_id = str(uuid.uuid5(uuid.NAMESPACE_URL, VERSION + ":" + fingerprint))
        target = self.root / (network_id + ".json")
        network = {"network_id": network_id, "title": title.strip(), "scope": scope, "created_at": now(),
                   "status": "derived_draft", "version": VERSION, "run_ids": sorted(loaded), "counts": counts,
                   "coverage": coverage, "inference_report": inference_report, "fingerprint": fingerprint,
                   "partial_extraction": combined["partial_extraction"]}
        combined.update(network_id=network_id, read_only=True, coverage=coverage, inference_report=inference_report,
                        provenance={"origin_runs": origins, "id_policy": "Full run UUID namespace; original repository locators, quotation strings, hashes and Unicode offsets retained", "review_policy": "Origin decisions are audit metadata only. Combined candidates and new analytical connections require review."})
        with lock(target.with_suffix(".lock")):
            if target.exists():
                return self._read_snapshot(target)["network"]
            save(target, {"network": network, "graph": combined})
            return network


def shared_network_store() -> NetworkStore:
    from app.api import repository
    from app.repository.knowledge_pipeline.jobs import shared_store
    return NetworkStore(Path(repository.STORAGE_ROOT) / "graph_networks", shared_store())
