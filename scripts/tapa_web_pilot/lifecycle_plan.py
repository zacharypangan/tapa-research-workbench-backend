#!/usr/bin/env python3
"""Plan retention actions from a source registry without executing any action.

Only the registry is read. Declared paths are labels, never opened or checked.
An optional ``downstream_inventory`` maps the categories below to lists of paths
or objects with a ``path``. ``downstream_inventory_complete: true`` is accepted
only when all categories are explicitly present (including empty lists).
"""

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)
CATEGORIES = (
    "caches", "representations", "translations", "embeddings", "evidence",
    "assertions", "reviews", "releases",
)
LINK_FIELDS = ("related_input_files", "linked_input_files", "linked_inputs")


def parse_timestamp(value, field):
    """Require a complete ISO timestamp with an explicit UTC offset."""
    if not isinstance(value, str) or not TIMESTAMP.fullmatch(value):
        raise ValueError("%s must be a timezone-aware ISO timestamp" % field)
    # datetime normalizes an invalid offset such as +00:60; reject that first.
    if value[-1] != "Z":
        if int(value[-5:-3]) > 23 or int(value[-2:]) > 59:
            raise ValueError("%s has an invalid timezone offset" % field)
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValueError("%s is an invalid timestamp" % field) from exc


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON number: %s" % value)


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Nonfinite JSON number: %s" % value)
    return parsed


def read_registry(path):
    raw = Path(path).read_bytes()
    records = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
        parse_constant=_reject_constant, parse_float=_finite_float,
    )
    if not isinstance(records, list):
        raise ValueError("Registry must be a JSON list of source objects")
    return records, hashlib.sha256(raw).hexdigest()


def _path_entry(item, field, role):
    if isinstance(item, str):
        result = {"path": item}
    elif isinstance(item, dict):
        # Preserve declared input IDs, hashes and any other inventory metadata.
        result = dict(item)
    else:
        raise ValueError("%s entries must be paths or objects with a path" % field)
    if not isinstance(result.get("path"), str) or not result["path"].strip():
        raise ValueError("%s entries require a nonempty path" % field)
    return {"declared_input": result, "registry_field": field, "role": role,
            "presence": "not_checked"}


def declared_inventory(source):
    inputs = []
    for field, role in (("path", "source_input"), ("representation_path", "representation")):
        if field in source:
            inputs.append(_path_entry(source[field], field, role))
    for field in LINK_FIELDS:
        if field in source:
            if not isinstance(source[field], list):
                raise ValueError("%s must be a list" % field)
            inputs.extend(_path_entry(item, field, "linked_input") for item in source[field])
    downstream = source.get("downstream_inventory", {})
    if not isinstance(downstream, dict) or any(key not in CATEGORIES for key in downstream):
        raise ValueError("downstream_inventory must contain recognized inventory categories")
    complete = source.get("downstream_inventory_complete", False)
    if not isinstance(complete, bool):
        raise ValueError("downstream_inventory_complete must be a boolean")
    if complete and set(downstream) != set(CATEGORIES):
        raise ValueError("Complete downstream inventory must explicitly declare every category")
    categories = {}
    for category in CATEGORIES:
        entries = downstream.get(category, [])
        if not isinstance(entries, list):
            raise ValueError("downstream_inventory.%s must be a list" % category)
        categories[category] = {
            "declaration": "declared" if category in downstream else "not_declared",
            "items": [_path_entry(item, "downstream_inventory.%s" % category, category)
                      for item in entries],
        }
    return {
        "declared_inputs": inputs,
        "downstream_categories": categories,
        "downstream_completeness": "declared_complete_unverified" if complete else "incomplete",
        "discovery_performed": False,
        "note": "Paths are preserved as declared; no existence, contents or dependency checks performed.",
    }


def source_actions(status, expires_at):
    actions = [{
        "action": "inventory_affected_materials",
        "execution": "not_performed",
        "categories": list(CATEGORIES),
        "instruction": "Trace source-dependent copies and products, including prompts/logs, translations, embeddings, evidence, assertions, reviews and release exports. Declared paths alone do not prove this inventory complete.",
    }]
    if status == "no_declared_expiry":
        actions.append({
            "action": "review_retention_basis",
            "instruction": "No expiry is declared. Review applicable source terms; this planner invents no TTL or renewal deadline.",
        })
        return actions
    actions.extend([
        {
            "action": "refresh_through_authorized_collector",
            "timing": "before_declared_expiry" if status == "active" else "due_now",
            "original_deadline": expires_at,
            "instruction": "Verify the collector/client and use-case authorization. Renew only resources successfully retrieved through the applicable documented collector; a manifest date edit or partial run does not renew other resources.",
        },
        {
            "action": "exclude_source_and_rebuild_if_unrefreshed",
            "timing": "at_declared_expiry" if status == "active" else "due_now",
            "instruction": "Exclude expired or unrefreshed source versions and their dependent claims, evidence, proposal metadata and review decisions from new releases. Rebuilding does not delete existing copies.",
        },
        {
            "action": "review_purge_necessity",
            "timing": "before_declared_expiry" if status == "active" else "due_now",
            "instruction": "Review deletion requirements for external caches, snapshots and all dependent products, including older releases and backups. Document and execute any required purge separately; this plan performs no deletion and grants no retention exception.",
        },
    ])
    return actions


def make_plan(records, as_of, registry_path, registry_sha256):
    cutoff = parse_timestamp(as_of, "as_of")
    seen = set()
    sources = []
    for source in records:
        if not isinstance(source, dict):
            raise ValueError("Every source must be an object")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip() or source_id != source_id.strip():
            raise ValueError("Every source requires a nonempty source_id without surrounding whitespace")
        if source_id in seen:
            raise ValueError("Duplicate source_id: %s" % source_id)
        seen.add(source_id)
        expiry = source.get("expires_at")
        deadline = parse_timestamp(expiry, "%s.expires_at" % source_id) if expiry is not None else None
        if deadline is None:
            status = "no_declared_expiry"
        elif cutoff < deadline:
            status = "active"
        else:
            status = "expired"
        sources.append({
            "source_id": source_id,
            "status": status,
            "due_at_cutoff": deadline is not None and cutoff == deadline,
            "expires_at": expiry,
            "expires_at_declared": "expires_at" in source,
            "deadline_utc": deadline.isoformat() if deadline else None,
            "graph_eligibility": source.get("graph_eligibility"),
            "eligibility_changed": False,
            "inventory": declared_inventory(source),
            "actions": source_actions(status, expiry),
        })
    states = ("active", "expired", "no_declared_expiry")
    return {
        "schema_version": "1.0.0",
        "execution_mode": "dry_run",
        "as_of": as_of,
        "as_of_utc": cutoff.isoformat(),
        "registry": {"declared_path": str(registry_path), "sha256": registry_sha256},
        "summary": {state: sum(item["status"] == state for item in sources) for state in states},
        "status_boundaries": {"active": "as_of < expires_at",
                              "expired": "as_of >= expires_at (including the exact deadline)",
                              "due_at_cutoff": "as_of == expires_at; also expired and subject to immediate actions",
                              "no_declared_expiry": "expires_at absent or null; no TTL inferred"},
        "limits": [
            "Only the supplied registry was read; declared input paths were not opened or checked.",
            "No network requests, credential access, cache changes, registry changes or deletions occurred.",
            "This planner does not enable derived extraction or grant platform permissions.",
            "Renewal requires resource-specific successful retrieval under applicable terms; manifest edits do not renew retention.",
            "Temporary retrieval errors and pagination absence do not establish resource deletion.",
            "Downstream inventory remains incomplete unless explicitly declared complete; declarations are unverified.",
        ],
        "sources": sources,
    }


def write_plan(registry, as_of, output):
    records, digest = read_registry(registry)
    plan = make_plan(records, as_of, registry, digest)
    serialized = json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output = Path(output)
    temporary_path = None
    try:
        # A private sibling ensures incomplete writes never appear at output.
        # Linking the completed file is atomic on POSIX and refuses an existing
        # destination, including a dangling symlink or a concurrent writer.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=".%s." % output.name, suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(str(temporary_path), str(output))
    finally:
        if temporary_path is not None:
            temporary_path.unlink()
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--as-of", required=True, help="Explicit timezone-aware cutoff, e.g. 2026-11-07T01:53:11Z")
    parser.add_argument("--output", required=True, type=Path, help="Fresh JSON output path; existing files are refused")
    args = parser.parse_args(argv)
    try:
        plan = write_plan(args.registry, args.as_of, args.output)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print("Dry-run lifecycle plan written for %d sources: %s" % (len(plan["sources"]), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
