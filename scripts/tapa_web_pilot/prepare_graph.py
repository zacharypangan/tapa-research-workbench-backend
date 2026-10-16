"""Validate an existing collection and prepare an UNBOUND graph review package.

This module uses only the standard library. It never collects URLs, imports browser
or parser dependencies, accepts curation bindings, or manufactures domain claims.
Run from stling_backend with ``python -m scripts.tapa_web_pilot.prepare_graph``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.parse import urlsplit


FORMAT_VERSION = 1
ELIGIBILITY = "quarantine_metadata_only"
RATIONALE = (
    "Parsing and checksum validation establish collection integrity only. An explicit "
    "curator check of source attribution, locator fidelity, rights and permitted "
    "transformation is required before evidence or semantic assertions are prepared."
)
ENGINES = {"static", "selenium", "playwright"}
EXTRACTORS = {"bs4", "trafilatura"}
TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)


class ValidationError(ValueError):
    """A collection does not satisfy the frozen collector interchange contract."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _constant(value):
    raise ValidationError(f"Non-finite JSON number: {value}")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"Non-finite JSON number: {value}")
    return number


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data: bytes, context: str):
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)
    except (UnicodeError, ValueError) as exc:
        raise ValidationError(f"{context}: invalid UTF-8 JSON ({exc})") from exc


def _rows(data: bytes, context: str) -> list[dict]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise ValidationError(f"{context}: invalid UTF-8") from exc
    result = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValidationError(f"{context}:{number}: blank JSONL row")
        row = _json(line.encode("utf-8"), f"{context}:{number}")
        if not isinstance(row, dict):
            raise ValidationError(f"{context}:{number}: expected an object")
        result.append(row)
    return result


def _require(row: dict, keys, context):
    if not isinstance(row, dict):
        raise ValidationError(f"{context}: expected an object")
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValidationError(f"{context}: missing {', '.join(missing)}")


def _string(value, context, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValidationError(f"{context}: expected a nonempty string")
    return value


def _url(value, context):
    _string(value, context)
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or "\\" in value
                or parsed.username is not None or parsed.password is not None
                or any(ord(char) <= 32 or ord(char) == 127 for char in value)):
            raise ValueError("expected an HTTP(S) URL without credentials or whitespace")
        parsed.port  # Validate malformed port syntax without making any network request.
    except ValueError as exc:
        raise ValidationError(f"{context}: invalid URL ({exc})") from exc
    return value


def _host(value):
    """Normalize the collector's host allowlist without DNS or any network access."""
    _string(value, "allowed_host")
    if any(char in value for char in "/@?#\\[]") or any(ord(char) <= 32 for char in value):
        raise ValidationError("manifest: invalid allowed host")
    try:
        normalized = value.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValidationError("manifest: invalid internationalized allowed host") from exc
    if ":" in normalized:
        try:
            ipaddress.IPv6Address(normalized)
        except ValueError as exc:
            raise ValidationError("manifest: invalid IPv6 allowed host") from exc
    if not normalized:
        raise ValidationError("manifest: empty allowed host")
    return normalized


def _timestamp(value, context):
    _string(value, context)
    if not TIMESTAMP.fullmatch(value):
        raise ValidationError(f"{context}: timezone required; expected a complete ISO timestamp")
    if not value.endswith("Z") and (int(value[-5:-3]) > 23 or int(value[-2:]) > 59):
        raise ValidationError(f"{context}: invalid timezone offset")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValidationError(f"{context}: invalid timestamp ({exc})") from exc


def _digest(value, context):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValidationError(f"{context}: expected a lowercase SHA-256")


def _choice(value, choices, context):
    if not isinstance(value, str) or value not in choices:
        raise ValidationError(f"{context}: unsupported value")


def _safe_file(root: Path, relative, context):
    _string(relative, context)
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValidationError(f"{context}: path must remain inside the collection")
    resolved = (root / Path(*path.parts)).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValidationError(f"{context}: missing file or symlink escapes the collection")
    return resolved


def validate_collection(collection: Path) -> dict:
    """Read and validate all inputs before creating any output directory."""
    root = collection.resolve()
    if not root.is_dir():
        raise ValidationError("Collection directory does not exist")
    manifest_bytes = _safe_file(root, "manifest.json", "manifest").read_bytes()
    manifest = _json(manifest_bytes, "manifest")
    _require(manifest, ["format_version", "created_at", "successful_sources", "attempted_sources",
                        "evidence_blocks", "statuses", "allowed_hosts", "requested_engine",
                        "requested_extractor", "seed_urls", "selectors", "limits", "files"], "manifest")
    if type(manifest["format_version"]) is not int or manifest["format_version"] != FORMAT_VERSION:
        raise ValidationError("Unsupported collector manifest format_version")
    created = _timestamp(manifest["created_at"], "manifest.created_at")
    _choice(manifest["requested_engine"], ENGINES | {"auto"}, "manifest.requested_engine")
    _choice(manifest["requested_extractor"], EXTRACTORS, "manifest.requested_extractor")
    if not isinstance(manifest["allowed_hosts"], list) or not manifest["allowed_hosts"]:
        raise ValidationError("manifest: allowed_hosts must be a nonempty list")
    allowed_hosts = {_host(host) for host in manifest["allowed_hosts"]}
    if len(set(manifest["allowed_hosts"])) != len(manifest["allowed_hosts"]):
        raise ValidationError("manifest: duplicate allowed host")
    if not isinstance(manifest["seed_urls"], list) or not manifest["seed_urls"]:
        raise ValidationError("manifest: seed_urls must be a nonempty list")
    for url in manifest["seed_urls"]:
        _url(url, "seed URL")
        if _host(urlsplit(url).hostname) not in allowed_hosts:
            raise ValidationError("manifest: seed hostname not allowed")
    _require(manifest["selectors"], ["content", "wait"], "manifest.selectors")
    if manifest["selectors"]["content"] is not None:
        _string(manifest["selectors"]["content"], "content selector", allow_empty=True)
    _string(manifest["selectors"]["wait"], "wait selector")
    _require(manifest["limits"], ["max_pages", "max_depth", "delay", "timeout", "response_bytes"], "manifest.limits")
    for key in ("max_pages", "max_depth", "response_bytes"):
        if type(manifest["limits"][key]) is not int or manifest["limits"][key] < (0 if key == "max_depth" else 1):
            raise ValidationError(f"manifest.limits: invalid {key}")
    for key in ("delay", "timeout"):
        value = manifest["limits"][key]
        if type(value) not in {int, float} or value < 0 or (key == "timeout" and value == 0):
            raise ValidationError(f"manifest.limits: invalid {key}")
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != {"sources.jsonl", "evidence_blocks.jsonl"}:
        raise ValidationError("manifest.files must declare exactly the collector JSONL files")
    input_bytes = {"manifest.json": manifest_bytes}
    for name in ("sources.jsonl", "evidence_blocks.jsonl"):
        _require(manifest["files"][name], ["sha256"], f"manifest.files.{name}")
        digest = manifest["files"][name]["sha256"]
        _digest(digest, f"manifest.files.{name}")
        data = _safe_file(root, name, name).read_bytes()
        if sha256(data) != digest:
            raise ValidationError(f"{name}: manifest checksum mismatch")
        input_bytes[name] = data
    sources = _rows(input_bytes["sources.jsonl"], "sources.jsonl")
    blocks = _rows(input_bytes["evidence_blocks.jsonl"], "evidence_blocks.jsonl")
    if any(not isinstance(row.get("status"), str) for row in sources):
        raise ValidationError("source: collection status must be a string")
    for key, actual in (("attempted_sources", len(sources)), ("evidence_blocks", len(blocks)),
                        ("successful_sources", sum(row.get("status") == "collected" for row in sources))):
        if type(manifest[key]) is not int or manifest[key] != actual:
            raise ValidationError(f"manifest: {key} count mismatch")
    if not isinstance(manifest["statuses"], dict) or any(type(count) is not int or count <= 0 for count in manifest["statuses"].values()):
        raise ValidationError("manifest: invalid status counts")
    if manifest["statuses"] != dict(Counter(row.get("status") for row in sources)):
        raise ValidationError("manifest: status counts mismatch")
    if len(sources) > manifest["limits"]["max_pages"]:
        raise ValidationError("manifest: attempt count exceeds page budget")
    by_source, snapshot_bytes, seen_hashes, discovered_pages = {}, {}, {}, set()
    for row in sources:
        _require(row, ["source_id", "requested_url", "retrieved_at", "depth", "discovered_from", "review_status",
                       "requested_engine", "requested_extractor", "status"], "source")
        sid = _string(row["source_id"], "source_id")
        if sid in by_source:
            raise ValidationError(f"{sid}: duplicate source ID")
        url = _url(row["requested_url"], f"{sid}.requested_url")
        if sid != "web-" + sha256(url.encode("utf-8"))[:20]:
            raise ValidationError(f"{sid}: source ID does not bind requested URL")
        if _host(urlsplit(url).hostname) not in allowed_hosts:
            raise ValidationError(f"{sid}: requested hostname not allowed")
        retrieved = _timestamp(row["retrieved_at"], f"{sid}.retrieved_at")
        if retrieved > created:
            raise ValidationError(f"{sid}: retrieval occurs after manifest creation")
        if type(row["depth"]) is not int or not 0 <= row["depth"] <= manifest["limits"]["max_depth"]:
            raise ValidationError(f"{sid}: invalid crawl depth")
        if row["discovered_from"] is not None:
            _url(row["discovered_from"], f"{sid}.discovered_from")
        if row["depth"] == 0 and (row["discovered_from"] is not None or url not in manifest["seed_urls"]):
            raise ValidationError(f"{sid}: depth-zero attempt must bind a seed URL")
        if row["depth"] > 0 and row["discovered_from"] not in discovered_pages:
            raise ValidationError(f"{sid}: crawl parent is not an earlier collected page")
        if (row["review_status"] != "needs_review" or row["requested_engine"] != manifest["requested_engine"]
                or row["requested_extractor"] != manifest["requested_extractor"]):
            raise ValidationError(f"{sid}: collector configuration/review status mismatch")
        if row["status"] not in {"collected", "skipped", "error"}:
            raise ValidationError(f"{sid}: unknown collection status")
        if row["status"] != "collected":
            _string(row.get("reason"), f"{sid}.reason")
            if any(key in row for key in ("snapshot", "content_sha256", "source_version_id")):
                raise ValidationError(f"{sid}: unsuccessful attempt has successful-source bindings")
            by_source[sid] = row
            continue
        _require(row, ["final_url", "http_status", "browser_http_status", "source_version_id", "captured_at",
                       "engine", "representation", "content_sha256", "snapshot", "duplicate_of", "redirect_chain",
                       "title", "language", "canonical_url_metadata", "published_at_metadata", "extraction_method",
                       "text", "structured_data", "warnings"], sid)
        _url(row["final_url"], f"{sid}.final_url")
        if _host(urlsplit(row["final_url"]).hostname) not in allowed_hosts:
            raise ValidationError(f"{sid}: final hostname not allowed")
        captured = _timestamp(row["captured_at"], f"{sid}.captured_at")
        if not retrieved <= captured <= created:
            raise ValidationError(f"{sid}: capture/retrieval chronology mismatch")
        if (type(row["http_status"]) is not int or row["http_status"] != 200
                or row["browser_http_status"] is not None and
                (type(row["browser_http_status"]) is not int or row["browser_http_status"] != 200)):
            raise ValidationError(f"{sid}: successful source has an unsuccessful HTTP status")
        if (not isinstance(row["engine"], str) or row["engine"] not in ENGINES
                or row["requested_engine"] != "auto" and row["engine"] != row["requested_engine"]):
            raise ValidationError(f"{sid}: invalid actual engine")
        expected_representation = "http_html" if row["engine"] == "static" else "rendered_dom"
        if row["representation"] != expected_representation or row["extraction_method"] != "beautifulsoup_dom_blocks":
            raise ValidationError(f"{sid}: engine/representation/extraction provenance mismatch")
        _digest(row["content_sha256"], f"{sid}.content_sha256")
        if row["source_version_id"] != sid + ":" + row["content_sha256"]:
            raise ValidationError(f"{sid}: source version does not bind content hash")
        data = _safe_file(root, row["snapshot"], f"{sid}.snapshot").read_bytes()
        if sha256(data) != row["content_sha256"]:
            raise ValidationError(f"{sid}: snapshot checksum mismatch")
        try:
            data.decode("utf-8")
        except UnicodeError as exc:
            raise ValidationError(f"{sid}: snapshot is not UTF-8") from exc
        if len(data) > manifest["limits"]["response_bytes"]:
            raise ValidationError(f"{sid}: snapshot exceeds recorded byte budget")
        if row["duplicate_of"] != seen_hashes.get(row["content_sha256"]):
            raise ValidationError(f"{sid}: duplicate_of does not bind an earlier equal snapshot")
        seen_hashes.setdefault(row["content_sha256"], sid)
        if not isinstance(row["warnings"], list) or any(not isinstance(item, str) for item in row["warnings"]):
            raise ValidationError(f"{sid}: malformed warning list")
        if not isinstance(row["structured_data"], list) or any(not isinstance(item, (dict, list)) for item in row["structured_data"]):
            raise ValidationError(f"{sid}: malformed reported structured data")
        for field in ("title", "language", "canonical_url_metadata", "published_at_metadata"):
            if row[field] is not None and not isinstance(row[field], str):
                raise ValidationError(f"{sid}: malformed reported {field}")
        _string(row["text"], f"{sid}.text")
        if ("article_text" in row) != ("article_text_method" in row):
            raise ValidationError(f"{sid}: incomplete supplementary article-text provenance")
        if "article_text" in row and (row["article_text_method"] != "trafilatura_supplementary"
                or row["article_text"] is not None and not isinstance(row["article_text"], str)
                or row["requested_extractor"] != "trafilatura"):
            raise ValidationError(f"{sid}: invalid supplementary article-text provenance")
        if not isinstance(row["redirect_chain"], list) or not row["redirect_chain"]:
            raise ValidationError(f"{sid}: missing HTTP redirect provenance")
        for step in row["redirect_chain"]:
            _require(step, ["url", "status"], f"{sid}.redirect_chain")
            _url(step["url"], f"{sid}.redirect_chain.url")
            if _host(urlsplit(step["url"]).hostname) not in allowed_hosts:
                raise ValidationError(f"{sid}: redirect-chain hostname not allowed")
            if type(step["status"]) is not int or step["status"] not in {200, 301, 302, 303, 307, 308}:
                raise ValidationError(f"{sid}: invalid redirect-chain status")
        if row["redirect_chain"][0]["url"] != url or row["redirect_chain"][-1]["status"] != 200:
            raise ValidationError(f"{sid}: redirect chain does not bind the request/success")
        if any(step["status"] == 200 for step in row["redirect_chain"][:-1]):
            raise ValidationError(f"{sid}: redirect chain contains a success before the final response")
        if row["engine"] == "static" and row["redirect_chain"][-1]["url"] != row["final_url"]:
            raise ValidationError(f"{sid}: static final URL disagrees with HTTP redirect chain")
        by_source[sid], snapshot_bytes[sid] = row, data
        discovered_pages.add(row["final_url"])
        input_bytes[row["snapshot"]] = data
    by_block, grouped = {}, defaultdict(list)
    for row in blocks:
        _require(row, ["block_id", "text", "locator", "tag", "evidence_id", "source_id", "source_version_id",
                       "source_url", "captured_at", "source_content_sha256", "review_status"], "block")
        eid = _string(row["evidence_id"], "evidence_id")
        if eid in by_block:
            raise ValidationError(f"{eid}: duplicate evidence ID")
        _string(row["source_id"], f"{eid}.source_id")
        source = by_source.get(row["source_id"])
        if not source or source["status"] != "collected":
            raise ValidationError(f"{eid}: unknown or unsuccessful source reference")
        _string(row["text"], f"{eid}.text")
        _string(row["locator"], f"{eid}.locator")
        _string(row["tag"], f"{eid}.tag")
        block_hash = sha256(json.dumps([row["text"], row["locator"]], ensure_ascii=False).encode("utf-8"))
        if row["block_id"] != block_hash or eid != source["source_version_id"] + ":" + block_hash:
            raise ValidationError(f"{eid}: block/evidence ID does not bind exact text and locator")
        for block_key, source_key in (("source_version_id", "source_version_id"), ("source_url", "final_url"),
                                      ("captured_at", "captured_at"), ("source_content_sha256", "content_sha256")):
            if row[block_key] != source[source_key]:
                raise ValidationError(f"{eid}: mismatched {block_key} binding")
        if row["review_status"] != "needs_review":
            raise ValidationError(f"{eid}: collection must not claim a completed source check")
        by_block[eid] = row
        grouped[row["source_id"]].append(row)
    for sid, source in by_source.items():
        if source["status"] == "collected" and source["text"] != "\n\n".join(row["text"] for row in grouped[sid]):
            raise ValidationError(f"{sid}: collected text differs from ordered DOM evidence blocks")
    return {"manifest": manifest, "sources": sources, "blocks": blocks, "grouped": grouped,
            "snapshot_bytes": snapshot_bytes, "input_bytes": input_bytes}


def _encoded(value, pretty=False):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       indent=2 if pretty else None) + "\n").encode("utf-8")


def _jsonl(rows):
    return b"".join(_encoded(row) for row in rows)


def prepare(collection: Path, output: Path, prefix: str = "WEB") -> dict:
    """Create a deterministic candidate package; never write the active KG registry."""
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,31}", prefix):
        raise ValidationError("Prefix must be 1–32 uppercase letters, digits or underscores, starting with a letter")
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValidationError("Choose a new output directory; existing outputs are not replaced")
    validated = validate_collection(collection)
    sources = sorted(validated["sources"], key=lambda row: row["source_id"])
    mapping = {row["source_id"]: prefix + "_" + row["source_id"][4:]
               for row in sources if row["status"] == "collected"}
    files, registry, bindings, candidates, coverage, ledger, lineage = {}, [], [], [], [], [], []
    for row in sources:
        sid = mapping.get(row["source_id"])
        ledger.append({"collector_source_id": row["source_id"], "proposed_source_id": sid,
                       "collection_status": row["status"], "collector_record": row,
                       "semantic_eligibility": "pending_curator_source_and_rights_check" if sid else "not_collected",
                       "verification": "collection_integrity_checked_only"})
        if sid is None:
            continue
        snapshot_path = f"snapshots/{sid}.html"
        metadata_path = f"source_text/{sid}.metadata.json"
        representation_path = f"source_text/{sid}.blocks.jsonl"
        files[snapshot_path] = validated["snapshot_bytes"][row["source_id"]]
        files[metadata_path] = _encoded(row, pretty=True)
        metadata_hash = sha256(files[metadata_path])
        source_candidates = []
        for index, block in enumerate(validated["grouped"][row["source_id"]], 1):
            candidate = {**block, "candidate_id": sid + ":candidate:" + block["block_id"],
                         "proposed_source_id": sid, "collector_block_index": index,
                         "verification": "pending_source_check", "text_kind": "collector_normalized_text",
                         "semantic_eligibility": "pending_curator_source_and_rights_check",
                         "provenance_kind": "dom_block", "source_metadata_path": metadata_path,
                         "source_metadata_sha256": metadata_hash}
            source_candidates.append(candidate)
            candidates.append(candidate)
        files[representation_path] = _jsonl(source_candidates)
        unit_page = sid + ":coverage:page-1"
        unit_blocks = [sid + ":coverage:block-" + block["block_id"] for block in source_candidates]
        source_hash, representation_hash = sha256(files[snapshot_path]), sha256(files[representation_path])
        registry.append({"source_id": sid, "path": str(output / snapshot_path), "expected_sha256": source_hash,
                         "source_kind": "captured_web_page", "representation_path": representation_path,
                         "representation_sha256": representation_hash, "representation_metadata_path": metadata_path,
                         "representation_metadata_sha256": metadata_hash,
                         "related_input_files": [{"input_id": sid + ":capture_metadata",
                                                  "path": str(output / metadata_path),
                                                  "expected_sha256": metadata_hash}], "title": row["title"],
                         "authors": None, "publication_year": None,
                         "published_at_metadata": row["published_at_metadata"], "captured_at": row["captured_at"],
                         "retrieved_at": row["retrieved_at"], "requested_url": row["requested_url"],
                         "source_url": row["final_url"], "canonical_url_metadata": row["canonical_url_metadata"],
                         "source_family_id": row["source_id"], "collector_source_id": row["source_id"],
                         "collector_source_version_id": row["source_version_id"],
                         "acquisition_method": row["engine"], "requested_engine": row["requested_engine"],
                         "capture_representation": row["representation"], "extraction_method": row["extraction_method"],
                         "graph_eligibility": ELIGIBILITY, "semantic_eligibility_rationale": RATIONALE,
                         "independent_observation_status": "not_established", "binding_status": "UNBOUND",
                         "expected_units": {"web_page": 1, "web_block": len(source_candidates)},
                         "expected_unit_ids": {"web_page": [unit_page], "web_block": unit_blocks},
                         "coverage_registry_complete": True,
                         "notes": "Candidate registration only. Capture time is not publication time. Reported metadata, "
                                  "supplementary article text and equal-snapshot proposals are not semantic claims."})
        bindings.append({"source_id": sid, "source_sha256": source_hash,
                         "representation_sha256": representation_hash, "binding_status": "UNBOUND",
                         "required_action": "Explicit curator source/rights check, source verification and version binding"})
        for unit_id, unit_type, locator in [(unit_page, "web_page", {"web_page": 1, "source_url": row["final_url"],
                                                                   "captured_at": row["captured_at"]})] + [
            (uid, "web_block", {"web_page": 1, "block_id": block["block_id"], "css_locator": block["locator"],
                                "collector_evidence_id": block["evidence_id"]})
            for uid, block in zip(unit_blocks, source_candidates)]:
            coverage.append({"unit_id": unit_id, "source_id": sid, "unit_type": unit_type, "locator": locator,
                             "disposition": "no_supported_assertion", "evidence_ids": [], "assertion_ids": [],
                             "notes": "Integrity checked and accounted for; source, rights and locator checks remain pending. "
                                      "No semantic extraction or completeness claim."})
        lineage.append({"proposed_source_id": sid, "collector_source_id": row["source_id"],
                        "collector_source_version_id": row["source_version_id"], "snapshot_path": snapshot_path,
                        "representation_path": representation_path, "source_metadata_path": metadata_path,
                        "source_metadata_sha256": metadata_hash,
                        "observed_page_url": row["final_url"], "captured_at": row["captured_at"],
                        "actual_engine": row["engine"], "capture_representation": row["representation"],
                        "dom_extraction_method": row["extraction_method"],
                        "supplementary_article_text_method": row.get("article_text_method"),
                        "supplementary_text_alignment": "not_source_aligned" if "article_text" in row else "not_present",
                        "duplicate_of_collector_source_id": row["duplicate_of"],
                        "duplicate_of_proposed_source_id": mapping.get(row["duplicate_of"]),
                        "duplicate_status": "equal_snapshot_candidate_only_no_identity_merge" if row["duplicate_of"] else "not_reported",
                        "verification": "pending_source_check", "graph_eligibility": ELIGIBILITY})
    files.update({"registry_proposal.json": _encoded(registry, pretty=True),
                  "binding_proposal.json": _encoded({"proposal_status": "UNBOUND", "not_an_accepted_curation_binding": True,
                                                     "semantic_eligibility_rationale": RATIONALE, "candidates": bindings}, pretty=True),
                  "source_id_mapping.json": _encoded(mapping, pretty=True),
                  "coverage_proposal.jsonl": _jsonl(coverage), "candidate_blocks.jsonl": _jsonl(candidates),
                  "attempt_ledger.jsonl": _jsonl(ledger), "source_lineage.jsonl": _jsonl(lineage),
                  "entities.jsonl": b"", "evidence.jsonl": b"", "assertions.jsonl": b"", "reviews.jsonl": b""})
    for name in ("manifest.json", "sources.jsonl", "evidence_blocks.jsonl"):
        files["acquisition/" + name] = validated["input_bytes"][name]
    manifest = {"format_version": FORMAT_VERSION, "package_kind": "unbound_web_graph_review_candidates",
                "collection_created_at": validated["manifest"]["created_at"], "prefix": prefix,
                "graph_eligibility": ELIGIBILITY, "semantic_eligibility_rationale": RATIONALE,
                "counts": {"attempts": len(sources), "registered_source_candidates": len(registry),
                           "candidate_blocks": len(candidates), "coverage_units": len(coverage),
                           "entities": 0, "semantic_evidence": 0, "assertions": 0, "reviews": 0},
                "collection_statuses": validated["manifest"]["statuses"],
                "input_files": {name: {"sha256": sha256(data)} for name, data in sorted(validated["input_bytes"].items())},
                "output_files": {name: {"sha256": sha256(data)} for name, data in sorted(files.items())},
                "preparer_sha256": sha256(Path(__file__).read_bytes()),
                "checks": {"collector_file_hashes": True, "snapshot_hashes": True, "block_digests": True,
                           "reference_bindings": True, "status_counts": True, "source_locator_verified": False,
                           "source_attribution_verified": False, "rights_verified": False, "semantic_recall_measured": False},
                "replication_note": "All non-location-bound outputs are deterministic. Registry absolute snapshot and metadata paths "
                                    "and their output checksum in this manifest vary with the chosen output directory. "
                                    "This manifest cannot contain its own output checksum."}
    files["preparation_manifest.json"] = _encoded(manifest, pretty=True)
    # Staging is outside the result. Validate fully before even creating its parent.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepare-graph-", dir=output.parent))
    reserved = False
    try:
        temporary.chmod(0o700)
        for relative, data in sorted(files.items()):
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.write_bytes(data)
            path.chmod(0o600)
        output.mkdir(mode=0o700)  # Exclusive reservation: do not replace even an empty existing output.
        reserved = True
        os.replace(temporary, output)
        reserved = False
    except FileExistsError as exc:
        raise ValidationError("Choose a new output directory; existing outputs are not replaced") from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if reserved:
            output.rmdir()  # Reservation remains empty if atomic replacement failed.
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="WEB")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.collection, args.output, args.prefix)
    except (ValidationError, OSError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")
    print(json.dumps({"package_kind": result["package_kind"], "counts": result["counts"],
                      "output": str(args.output.absolute())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
