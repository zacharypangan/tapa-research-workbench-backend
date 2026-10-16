"""Enumerate a local JSON snapshot as UNBOUND structural review candidates.

No network requests, HTML parsing, domain claims, source eligibility decisions or
registry changes occur. JSON strings (including embedded HTML) remain literal
fields. An omitted record-list pointer treats the complete root as one record;
an explicit RFC6901 pointer must select an array of record objects. Every leaf
outside that array is still emitted, with no invented record association.
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import re
import shutil
import tempfile

try:
    from . import prepare_graph as strict
except ImportError:
    import prepare_graph as strict


ValidationError = strict.ValidationError


def pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def resolve_pointer(value, pointer: str):
    """Resolve an RFC6901 JSON Pointer; URI fragments and '-' are not indexes."""
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValidationError("record_list_pointer must be an RFC6901 JSON Pointer")
    if not pointer:
        return value
    current = value
    for encoded in pointer.split("/")[1:]:
        if re.search(r"~(?:[^01]|$)", encoded):
            raise ValidationError("record_list_pointer contains an invalid escape")
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise ValidationError("record_list_pointer does not exist")
            current = current[token]
        elif isinstance(current, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token) or int(token) >= len(current):
                raise ValidationError("record_list_pointer contains an invalid array index")
            current = current[int(token)]
        else:
            raise ValidationError("record_list_pointer traverses a scalar")
    return current


def json_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    raise ValidationError("Unsupported parsed JSON value")


def _locator_id(version, kind, pointer):
    return version + ":" + kind + ":" + strict.sha256(pointer.encode("utf-8"))


def _enumerate(value, source, selected_pointer):
    version = source["source_version_id"]
    if selected_pointer is None:
        record_pointers = [""]
    else:
        selected = resolve_pointer(value, selected_pointer)
        if not isinstance(selected, list) or any(not isinstance(item, dict) for item in selected):
            raise ValidationError("record_list_pointer must select an array of objects")
        record_pointers = [selected_pointer + "/" + str(index) for index in range(len(selected))]
    record_ids = {pointer: _locator_id(version, "record", pointer) for pointer in record_pointers}
    fields, objects = [], []

    def record_for(pointer):
        if selected_pointer is None:
            return record_ids[""]
        prefix = selected_pointer + "/"
        if pointer.startswith(prefix):
            index = pointer[len(prefix):].split("/", 1)[0]
            return record_ids.get(prefix + index)
        return None

    def visit(node, pointer, parent_object=None):
        kind = json_type(node)
        object_id = parent_object
        if kind == "object":
            object_id = _locator_id(version, "object", pointer)
            objects.append({
                "object_association_id": object_id, "json_pointer": pointer,
                "parent_object_association_id": parent_object,
                "record_id": record_for(pointer), "source_id": source["source_id"],
                "source_version_id": version,
                "association_kind": "nested_json_structure_only",
            })
        if kind in {"object", "array"} and node:
            children = sorted(node.items()) if kind == "object" else enumerate(node)
            for key, child in children:
                visit(child, pointer + "/" + pointer_token(str(key)), object_id)
            return
        value_type = "empty_" + kind if kind in {"object", "array"} else kind
        candidate_id = _locator_id(version, "field", pointer)
        fields.append({
            "candidate_id": candidate_id,
            "candidate_version_id": candidate_id + ":" + source["metadata_sha256"],
            "source_id": source["source_id"], "source_version_id": version,
            "snapshot_sha256": source["snapshot_sha256"],
            "source_metadata_sha256": source["metadata_sha256"],
            "source_url": source["source_url"], "captured_at": source["captured_at"],
            "locator": {"kind": "rfc6901_json_pointer", "json_pointer": pointer,
                        "snapshot_path": "snapshot.json"},
            "record_id": record_for(pointer), "object_association_id": object_id,
            "value_type": value_type, "value": node,
            "interpretation": "uninterpreted_json_field",
            "review_status": "needs_source_rights_and_semantic_mapping_review",
            "graph_eligibility": "quarantine_metadata_only",
        })

    visit(value, "")
    record_counts = Counter(field["record_id"] for field in fields if field["record_id"] is not None)
    records = [{
        "record_id": record_ids[pointer], "record_pointer": pointer,
        "source_id": source["source_id"], "source_version_id": version,
        "record_type": json_type(resolve_pointer(value, pointer)),
        "structural_leaf_count": record_counts[record_ids[pointer]],
        "association_kind": "selected_json_record_only",
    } for pointer in record_pointers]
    return fields, objects, records


def extract(snapshot: Path, source_url: str, captured_at: str, expected_sha256: str,
            output: Path, record_list_pointer: str | None = None) -> dict:
    """Validate completely, then publish a fresh, pinned local review package."""
    snapshot, output = Path(snapshot), Path(output)
    strict._url(source_url, "source_url")  # Syntax only; no DNS/network access.
    captured = strict._timestamp(captured_at, "captured_at")
    strict._digest(expected_sha256, "expected_sha256")
    raw = snapshot.read_bytes()
    digest = strict.sha256(raw)
    if digest != expected_sha256:
        raise ValidationError("Snapshot SHA-256 does not match expected_sha256")
    value = strict._json(raw, "snapshot")
    source_id = "json-" + strict.sha256(source_url.encode("utf-8"))[:20]
    metadata = {
        "format_version": 1, "source_id": source_id,
        "source_version_id": source_id + ":" + digest,
        "source_url": source_url, "captured_at": captured_at,
        "captured_at_utc": captured.isoformat(), "snapshot_path": "snapshot.json",
        "snapshot_sha256": digest, "snapshot_bytes": len(raw),
        "acquisition_method": "provided_local_json_snapshot",
        "acquisition_attribution": "URL and capture time explicitly supplied by caller; not independently verified",
        "network_verification_performed": False, "http_status": None,
        "representation": "typed_structural_json_fields",
        "record_list_pointer": record_list_pointer,
        "graph_eligibility": "quarantine_metadata_only", "binding_status": "UNBOUND",
        "rights_review": "pending", "semantic_mapping_review": "pending",
        "string_handling": "Verbatim parsed JSON strings; embedded HTML is not parsed or promoted to prose evidence",
        "immutability_policy": "Original bytes copied read-only and content hash pinned; this is not a WORM storage guarantee",
    }
    metadata_bytes = strict._encoded(metadata, pretty=True)
    source = dict(metadata, metadata_sha256=strict.sha256(metadata_bytes))
    fields, objects, records = _enumerate(value, source, record_list_pointer)
    counts = {
        "structural_leaves": len(fields), "records": len(records),
        "object_associations": len(objects),
        "scalar_fields": sum(field["value_type"] not in {"empty_object", "empty_array"} for field in fields),
        "empty_container_fields": sum(field["value_type"] in {"empty_object", "empty_array"} for field in fields),
        "fields_outside_selected_records": sum(field["record_id"] is None for field in fields),
        "value_types": dict(sorted(Counter(field["value_type"] for field in fields).items())),
        "semantic_entities": 0, "semantic_evidence": 0, "semantic_assertions": 0, "human_reviews": 0,
    }
    coverage = [{
        "source_id": source_id, "source_version_id": source["source_version_id"],
        "unit_kind": "json_structural_leaf", "json_pointer": field["locator"]["json_pointer"],
        "candidate_id": field["candidate_id"], "disposition": "structural_candidate_only",
    } for field in fields] + [{
        "source_id": source_id, "source_version_id": source["source_version_id"],
        "unit_kind": "json_selected_record", "json_pointer": record["record_pointer"],
        "record_id": record["record_id"], "disposition": "structural_record_accounted",
    } for record in records]
    files = {
        "snapshot.json": raw, "source_metadata.json": metadata_bytes,
        "candidate_fields.jsonl": strict._jsonl(fields), "record_inventory.jsonl": strict._jsonl(records),
        "object_associations.jsonl": strict._jsonl(objects), "coverage.jsonl": strict._jsonl(coverage),
        "entities.jsonl": b"", "evidence.jsonl": b"", "assertions.jsonl": b"", "reviews.jsonl": b"",
    }
    manifest = {
        "format_version": 1, "package_kind": "unbound_json_structural_candidates",
        "source_id": source_id, "source_version_id": source["source_version_id"],
        "source_metadata_sha256": source["metadata_sha256"], "counts": counts,
        "input_snapshot_sha256": digest,
        "extractor_sha256": strict.sha256(Path(__file__).read_bytes()),
        "strict_helpers_sha256": strict.sha256(Path(strict.__file__).read_bytes()),
        "files": {name: {"sha256": strict.sha256(data), "bytes": len(data)} for name, data in sorted(files.items())},
        "limits": [
            "Structural leaf accounting is exhaustive for this validated snapshot, not a measure of semantic recall.",
            "Record/object IDs are JSON structure associations, not museum objects, people or archaeological entity identities.",
            "JSON strings are preserved without HTML or prose extraction; source rights and semantic mapping remain pending.",
            "No network, platform approval, registry binding, semantic evidence, assertion or human review is created.",
            "Raw source bytes remain authoritative for numeric lexemes and formatting; typed candidates are parsed JSON representations.",
        ],
    }
    files["manifest.json"] = strict._encoded(manifest, pretty=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".extract-json-", dir=output.parent))
    reserved = False
    try:
        temporary.chmod(0o700)
        for name, data in sorted(files.items()):
            path = temporary / name
            path.write_bytes(data)
            path.chmod(0o400 if name == "snapshot.json" else 0o600)
        output.mkdir(mode=0o700)  # Refuse existing directories, files and symlinks.
        reserved = True
        os.replace(temporary, output)
        reserved = False
    except FileExistsError as exc:
        raise ValidationError("Choose a fresh output directory; existing outputs are not replaced") from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if reserved:
            output.rmdir()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--captured-at", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--record-list-pointer", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = extract(args.snapshot, args.source_url, args.captured_at, args.expected_sha256,
                           args.output, args.record_list_pointer)
    except (OSError, ValueError, TypeError, RecursionError) as error:
        parser.error(str(error))
    print(strict._encoded({"package_kind": manifest["package_kind"], "counts": manifest["counts"]}).decode().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
