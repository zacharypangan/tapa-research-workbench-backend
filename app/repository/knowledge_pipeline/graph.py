"""Compile evidence-bound candidate records into an immutable, local release.

This module does no acquisition or semantic inference. Inclusion decisions do not
constitute specialist acceptance of a statement's scientific truth.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class GraphValidationError(ValueError):
    """The candidate collection cannot safely be compiled."""


_REVIEW_FIELDS = {
    "candidate_version", "review_status", "status", "inclusion_review_status",
    "inclusion_decision", "review_decision", "reviewed_at", "review_actor_id",
}
_SOURCE_FIELDS = {
    "source_id", "source_kind", "kind", "origin", "role", "source_role",
    "title", "filename", "mime_type", "source_url", "url", "authors",
    "publication_date", "captured_at", "uploaded_at", "snapshot_sha256",
    "representation_sha256", "content_sha256", "sha256", "source_version_id",
    "representation_version_id", "seed_ids", "seeds", "source_family",
    "semantic_eligible", "eligible_for_graph", "retention_expires_at",
    "source_locator", "acquisition_method", "raw_sha256", "extractor", "status",
    "warnings", "representation_path", "snapshot_path", "extractor_code_sha256",
    "python_version", "snapshot_hash_basis", "unreadable_locators",
    "original_filename", "byte_count", "unit_count", "representation_kind",
    "requested_url", "questions", "collection_path",
}
_SCOPE_TYPES = {"place_id": {"Place", "FindContext"},
                "time_id": {"TimeExpression", "DatingAssessment"},
                "scheme_id": {"TypologyScheme"}, "production_case_id": {"ProductionCase"},
                "case_id": {"ProductionCase"}}


def _json(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GraphValidationError("Records must contain finite JSON values") from exc


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _need(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GraphValidationError(f"{name} must be a nonempty string")
    return value


def _index(rows: list, key: str) -> dict[str, dict]:
    result = {}
    if not isinstance(rows, list):
        raise GraphValidationError(f"{key} records must be a list")
    for row in rows:
        if not isinstance(row, dict):
            raise GraphValidationError(f"{key} record must be an object")
        rid = _need(row.get(key), key)
        if rid in result:
            raise GraphValidationError(f"Duplicate {key}: {rid}")
        _json(row)
        result[rid] = dict(row)
    return result


def _version(record: dict, binding: dict) -> str:
    body = {k: v for k, v in record.items() if k not in _REVIEW_FIELDS}
    return _sha(_json({"record": body, "bindings": binding}))


def _names(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {v if isinstance(v, str) else v.get("name", v.get("id")): v
                for v in value if isinstance(v, (str, dict))}
    return {}


def _source_role(source: dict) -> str:
    for field in ["role", "source_role", "origin"]:
        role = source.get(field)
        if role in {"primary", "web", "curation"}:
            return role
    kind = str(source.get("source_kind", source.get("kind", source.get("origin", "")))).lower()
    if kind in {"web", "webpage", "web_snapshot", "website"} or kind.startswith("web_"):
        return "web"
    if kind in {"primary", "upload", "document", "pdf", "text", "docx", "uploaded_document"}:
        return "primary"
    if kind in {"curation", "project_curation"}:
        return "curation"
    return "unknown"


def _ids(value: Any, name: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise GraphValidationError(f"{name} must be a {'nonempty ' if nonempty else ''}list")
    if any(not isinstance(v, str) or not v for v in value) or len(set(value)) != len(value):
        raise GraphValidationError(f"{name} must contain distinct nonempty IDs")
    return value


def _bound_refs(record: dict, sources: dict, evidence: dict, *, own_source: bool) -> list[str]:
    refs = _ids(record.get("evidence_ids"), "evidence_ids")
    if any(eid not in evidence for eid in refs):
        raise GraphValidationError("Unknown evidence reference")
    sid = record.get("source_id")
    if own_source:
        if sid not in sources:
            raise GraphValidationError("Unknown or missing source_id")
        if not any(evidence[eid]["source_id"] == sid for eid in refs):
            raise GraphValidationError("Statement/entity lacks evidence from its asserted source")
        source = sources[sid]
        if source.get("semantic_eligible") is False or source.get("eligible_for_graph") is False:
            raise GraphValidationError("Source is ineligible for semantic graph use")
        expected = record.get("source_version_id")
        if expected is not None and expected != source["source_version_id"]:
            raise GraphValidationError("Stale source version binding")
    bindings = record.get("evidence_version_ids", {})
    if not isinstance(bindings, dict):
        raise GraphValidationError("evidence_version_ids must be an object")
    for eid, version in bindings.items():
        if eid not in refs or version != evidence[eid]["evidence_version_id"]:
            raise GraphValidationError("Stale or unreferenced evidence version binding")
    return refs


def _ontology_check(record: dict, entities: dict, predicates: dict) -> None:
    predicate = _need(record.get("predicate"), "predicate")
    if predicate not in predicates:
        raise GraphValidationError(f"Undefined predicate: {predicate}")
    definition = predicates[predicate]
    if not isinstance(definition, dict):
        definition = {}
    subject = record.get("subject_id")
    obj = record.get("object_id")
    if subject not in entities or (obj is not None and obj not in entities):
        raise GraphValidationError("Unknown statement entity endpoint")
    if (obj is None) == ("value" not in record):
        raise GraphValidationError("Statement needs exactly one object_id or value")
    if definition.get("requires_object") and obj is None:
        raise GraphValidationError("Predicate requires an entity object")
    for key, eid in [("subject_types", subject), ("domain", subject),
                     ("object_types", obj), ("range", obj)]:
        if eid is not None and definition.get(key) and entities[eid]["entity_type"] not in definition[key]:
            raise GraphValidationError(f"Predicate {key} constraint failed")
    scope = record.get("scope", {})
    if not isinstance(scope, dict):
        raise GraphValidationError("scope must be an object")
    if definition.get("requires_case_scope") and "case_id" not in scope:
        raise GraphValidationError("Predicate requires explicit case scope")
    for key, types in _SCOPE_TYPES.items():
        eid = scope.get(key)
        if eid is not None and (eid not in entities or entities[eid]["entity_type"] not in types):
            raise GraphValidationError(f"Invalid scope reference: {key}")
    if record.get("polarity") not in {"positive", "negative", "uncertain"}:
        raise GraphValidationError("polarity must be positive, negative or uncertain")


def _entity_refs(record: dict) -> list[str]:
    """Context records are dependencies as well as the statement endpoints."""
    refs = [record["subject_id"]] + ([record["object_id"]] if record.get("object_id") else [])
    refs.extend(record.get("scope", {}).get(key) for key in _SCOPE_TYPES
                if record.get("scope", {}).get(key) is not None)
    return list(dict.fromkeys(refs))


def _reviews(decisions: list, candidates: dict) -> tuple[dict, list]:
    active, history = {}, []
    if not isinstance(decisions, list):
        raise GraphValidationError("decisions must be a list")
    for raw in decisions:
        if not isinstance(raw, dict):
            raise GraphValidationError("Review decision must be an object")
        d = dict(raw)
        cid = _need(d.get("candidate_id"), "candidate_id")
        if cid not in candidates:
            raise GraphValidationError("Review decision references unknown candidate")
        if d.get("decision") not in {"accept", "reject"}:
            raise GraphValidationError("Decision must be accept or reject")
        _need(d.get("actor_id"), "actor_id")
        _need(d.get("rationale"), "rationale")
        try:
            at = datetime.fromisoformat(_need(d.get("at"), "at").replace("Z", "+00:00"))
        except ValueError as exc:
            raise GraphValidationError("Invalid review timestamp") from exc
        if at.tzinfo is None or at.utcoffset() is None:
            raise GraphValidationError("Review timestamp must include a timezone")
        if d.get("candidate_version") not in {None, candidates[cid]["candidate_version"]}:
            raise GraphValidationError("Stale review candidate version")
        d["candidate_version"] = candidates[cid]["candidate_version"]
        previous = active.get(cid)
        if previous and previous[0] == at and previous[1]["decision"] != d["decision"]:
            raise GraphValidationError("Conflicting simultaneous review decisions")
        if previous is None or at > previous[0]:
            active[cid] = (at, d)
        history.append(d)
    return {cid: d for cid, (_, d) in active.items()}, history


def _ttl(graph: dict) -> str:
    def uri(kind, rid):
        return f"<urn:knowledge-pipeline:{kind}:{_sha(str(rid).encode('utf-8'))}>"
    def literal(value):
        return json.dumps(str(value), ensure_ascii=False)
    lines = ["@prefix kp: <urn:knowledge-pipeline:v1:> .", ""]
    for e in graph["entities"]:
        lines.append(f'{uri("entity", e["entity_id"])} a kp:EntityRecord ; kp:recordId {literal(e["entity_id"])} ; kp:entityType {literal(e["entity_type"])} .')
    for e in graph["evidence"]:
        lines.append(f'{uri("evidence", e["evidence_id"])} a kp:EvidenceFragment ; kp:source {uri("source", e["source_id"])} ; kp:locatorJSON {literal(_json(e["locator"]).decode().strip())} ; kp:text {literal(e["text"])} .')
    for s in graph["sources"]:
        lines.append(f'{uri("source", s["source_id"])} a kp:SourceRecord ; kp:recordId {literal(s["source_id"])} ; kp:version {literal(s["source_version_id"])} .')
    for layer, records, key in [("SourceClaim", graph["assertions"], "assertion_id"),
                                ("AnalyticalAlignment", graph["alignments"], "alignment_id")]:
        for c in records:
            values = [f'a kp:{layer}', f'kp:claimId {literal(c[key])}',
                      f'kp:subject {uri("entity", c["subject_id"])}',
                      f'kp:predicate {literal(c["predicate"])}',
                      f'kp:polarity {literal(c["polarity"])}',
                      f'kp:scopeJSON {literal(_json(c.get("scope", {})).decode().strip())}',
                      'kp:reviewStatus "needs_specialist_review"',
                      f'kp:inclusionStatus {literal(c["inclusion_review_status"])}',
                      f'kp:evidence {", ".join(uri("evidence", eid) for eid in c["evidence_ids"])}']
            if c.get("object_id") is not None:
                values.append(f'kp:object {uri("entity", c["object_id"])}')
            else:
                values.append(f'kp:valueJSON {literal(_json(c["value"]).decode().strip())}')
            if c.get("source_id"):
                values.append(f'kp:source {uri("source", c["source_id"])}')
            if c.get("scientific_status"):
                values.append(f'kp:scientificStatus {literal(c["scientific_status"])}')
            lines.append(f'{uri("claim", c[key])} ' + " ; ".join(values) + " .")
    return "\n".join(lines) + "\n"


def build_release(output_dir: Path, ontology: dict, sources: list, entities: list,
                  evidence: list, assertions: list, alignments: list,
                  decisions: list, allow_pending: bool = False) -> dict:
    """Validate and export a fresh preview or accepted-inclusion release.

    Pending/rejected candidates are excluded from an accepted release. An
    accepted claim with unaccepted entity endpoints is an error. Preview still
    excludes rejected records. No source path is opened and no URL is fetched.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Release directory already exists: {output_dir}")
    if not isinstance(ontology, dict):
        raise GraphValidationError("ontology must be an object")
    _need(ontology.get("version"), "ontology.version")
    ontology_sha = _sha(_json(ontology))
    types = _names(ontology.get("entity_types", ontology.get("types")))
    predicates = _names(ontology.get("predicates"))
    if not types or not predicates:
        raise GraphValidationError("Ontology must define entity types and predicates")
    src = _index(sources, "source_id")
    ent = _index(entities, "entity_id")
    ev = _index(evidence, "evidence_id")
    claims = _index(assertions, "assertion_id")
    analytical = _index(alignments, "alignment_id")
    if set(ent) & set(claims) or set(ent) & set(analytical) or set(claims) & set(analytical):
        raise GraphValidationError("Candidate IDs must be unique across record types")
    for sid, s in src.items():
        for field in ["snapshot_sha256", "representation_sha256", "content_sha256", "sha256", "raw_sha256"]:
            digest = s.get(field)
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                       or any(c not in "0123456789abcdef" for c in digest.lower())):
                raise GraphValidationError(f"Invalid source hash: {field}")
        clean = {k: v for k, v in s.items() if k in _SOURCE_FIELDS}
        clean["source_version_id"] = s.get("source_version_id") or f'{sid}:{_sha(_json(clean))}'
        _need(clean["source_version_id"], "source_version_id")
        src[sid] = clean
    for eid, e in ev.items():
        sid = e.get("source_id")
        if sid not in src:
            raise GraphValidationError("Evidence references unknown source")
        if not isinstance(e.get("locator"), (str, dict)) or not e["locator"]:
            raise GraphValidationError("Evidence requires an exact locator")
        if "text" not in e and isinstance(e.get("quote"), str):
            e["text"] = e["quote"]
        _need(e.get("text"), "evidence.text")
        if e.get("text_sha256") not in {None, _sha(e["text"].encode("utf-8"))}:
            raise GraphValidationError("Evidence text hash mismatch")
        if e.get("source_version_id") not in {None, src[sid]["source_version_id"]}:
            raise GraphValidationError("Stale evidence source version")
        for field in ["snapshot_sha256", "representation_sha256"]:
            if field in e and e[field] != src[sid].get(field):
                raise GraphValidationError("Stale evidence representation/hash binding")
        e["source_version_id"] = src[sid]["source_version_id"]
        e.setdefault("evidence_version_id", f'{eid}:{_version(e, {"source": src[sid]})}')
    for eid, e in ent.items():
        if e.get("entity_type") not in types:
            raise GraphValidationError("Undefined entity type")
        refs = _bound_refs(e, src, ev, own_source=True)
        e["candidate_version"] = _version(e, {"ontology": ontology_sha,
            "evidence": {rid: ev[rid]["evidence_version_id"] for rid in refs}})
    seed_bindings: dict[str, set[str]] = {}
    for eid, e in ent.items():
        if _source_role(src[e["source_id"]]) == "primary":
            seed_bindings.setdefault(eid, set()).add(eid)
            for seed in e.get("seed_ids", []):
                seed_bindings.setdefault(seed, set()).add(eid)
    for s in src.values():
        for seed in s.get("seeds", []):
            if isinstance(seed, dict):
                sid, eid = seed.get("seed_id"), seed.get("entity_id", seed.get("subject_id"))
                if isinstance(sid, str) and eid in ent and _source_role(src[ent[eid]["source_id"]]) == "primary":
                    seed_bindings.setdefault(sid, set()).add(eid)
    for kind, rows in [("assertion", claims), ("alignment", analytical)]:
        for cid, c in rows.items():
            _ontology_check(c, ent, predicates)
            refs = _bound_refs(c, src, ev, own_source=kind == "assertion")
            if c.get("ontology_version") not in {None, ontology["version"]}:
                raise GraphValidationError("Stale ontology version binding")
            if c.get("ontology_sha256") not in {None, ontology_sha}:
                raise GraphValidationError("Stale ontology hash binding")
            endpoints = _entity_refs(c)
            bindings = c.get("entity_version_ids", {})
            if not isinstance(bindings, dict):
                raise GraphValidationError("entity_version_ids must be an object")
            for eid, version in bindings.items():
                expected = ent.get(eid, {}).get("entity_version_id", ent.get(eid, {}).get("candidate_version"))
                if eid not in endpoints or version != expected:
                    raise GraphValidationError("Stale entity version binding")
            if kind == "alignment":
                if c["predicate"] != "aligns_reported_concept_with":
                    raise GraphValidationError("Analytical alignment requires its dedicated predicate")
                obj = c.get("object_id")
                if obj is None:
                    raise GraphValidationError("Alignment needs an entity object")
                primary_sid = ent[c["subject_id"]]["source_id"]
                web_sid = ent[obj]["source_id"]
                if _source_role(src[primary_sid]) != "primary" or _source_role(src[web_sid]) != "web":
                    raise GraphValidationError("Alignment must link a primary subject to a web object")
                seeds = c.get("seed_ids", c.get("scope", {}).get("seed_ids"))
                seeds = _ids(seeds, "alignment.seed_ids")
                scope_seeds = c.get("scope", {}).get("seed_ids")
                if scope_seeds is not None and set(_ids(scope_seeds, "scope.seed_ids")) != set(seeds):
                    raise GraphValidationError("Conflicting alignment seed bindings")
                web_seeds = _ids(src[web_sid].get("seed_ids", []), "web source.seed_ids", nonempty=False)
                if any(c["subject_id"] not in seed_bindings.get(seed, set()) or seed not in web_seeds for seed in seeds):
                    raise GraphValidationError("Alignment seed is not bound to this primary record and web source")
                if not any(ev[eid]["source_id"] == primary_sid for eid in refs) or not any(ev[eid]["source_id"] == web_sid for eid in refs):
                    raise GraphValidationError("Alignment needs both primary and web source evidence")
                if c.get("source_id") and (c["source_id"] not in src or _source_role(src[c["source_id"]]) != "curation"):
                    raise GraphValidationError("Analytical alignment must not impersonate an original source")
                c["seed_ids"] = seeds
                c["scientific_status"] = "project_only"
                c["attribution"] = "analytical_project_curation"
            c["candidate_version"] = _version(c, {"ontology": ontology_sha,
                "entities": {eid: ent[eid]["candidate_version"] for eid in endpoints},
                "evidence": {eid: ev[eid]["evidence_version_id"] for eid in refs}})
    candidates = {**ent, **claims, **analytical}
    active, history = _reviews(decisions, candidates)
    if candidates and not allow_pending and not any(d["decision"] == "accept" for d in active.values()):
        raise GraphValidationError("No accepted candidates; request a pending preview or review the records")
    def verdict(cid):
        return active.get(cid, {}).get("decision", "pending")
    def include(cid):
        return verdict(cid) != "reject" if allow_pending else verdict(cid) == "accept"
    included_ent = {eid: e for eid, e in ent.items() if include(eid)}
    selected_claims, selected_alignments, exclusions = [], [], []
    for kind, rows, selected in [("assertion", claims, selected_claims), ("alignment", analytical, selected_alignments)]:
        for cid, c in rows.items():
            endpoints = _entity_refs(c)
            if not include(cid):
                exclusions.append({"candidate_id": cid, "reason": verdict(cid)})
                continue
            if any(eid not in included_ent for eid in endpoints):
                if not allow_pending and verdict(cid) == "accept":
                    raise GraphValidationError("Accepted claim requires accepted entity endpoints")
                exclusions.append({"candidate_id": cid, "reason": "excluded_entity_endpoint"})
                continue
            selected.append(c)
    exclusions.extend({"candidate_id": eid, "reason": verdict(eid)} for eid in ent if eid not in included_ent)
    for cid, c in candidates.items():
        c["review_status"] = "needs_specialist_review"
        c["inclusion_review_status"] = "accepted_for_inclusion" if verdict(cid) == "accept" else verdict(cid)
    selected_evidence_ids = {eid for e in included_ent.values() for eid in e["evidence_ids"]}
    selected_evidence_ids.update(eid for c in selected_claims + selected_alignments for eid in c["evidence_ids"])
    selected_ev = [ev[eid] for eid in sorted(selected_evidence_ids)]
    counts = {"sources": len(src), "entities": len(included_ent), "evidence": len(selected_ev),
              "assertions": len(selected_claims), "alignments": len(selected_alignments),
              "statements": len(selected_claims) + len(selected_alignments), "review_decisions": len(history),
              "specialist_review_decisions": 0, "excluded_candidates": len(exclusions)}
    graph = {"format_version": 1, "release_kind": "pending_preview" if allow_pending else "accepted_inclusion",
             "ontology_version": ontology["version"], "ontology_sha256": ontology_sha,
             "entities": [included_ent[eid] for eid in sorted(included_ent)],
             "assertions": sorted(selected_claims, key=lambda c: c["assertion_id"]),
             "alignments": sorted(selected_alignments, key=lambda c: c["alignment_id"]),
             "evidence": selected_ev, "sources": [src[sid] for sid in sorted(src)],
             "decisions": history, "excluded_candidates": exclusions,
             "scientific_limits": {"inclusion_approval_is_specialist_truth": False,
                                   "precision": None, "recall": None, "annotator_agreement": None}}
    metrics = {"counts": counts, "release_kind": graph["release_kind"],
               "review_status": "needs_specialist_review", "scientific_evaluation": graph["scientific_limits"]}
    artifacts = {"graph.json": _json(graph), "ontology.json": _json(ontology), "metrics.json": _json(metrics),
                 "graph.ttl": _ttl(graph).encode("utf-8")}
    for key in ["entities", "assertions", "alignments", "evidence", "sources", "decisions"]:
        artifacts[key + ".jsonl"] = b"".join(_json(row) for row in graph[key])
    hashes = {name: _sha(data) for name, data in artifacts.items()}
    manifest = {"format_version": 1, "release_kind": graph["release_kind"], "counts": counts,
                "ontology_sha256": ontology_sha, "artifact_sha256": hashes,
                "hash_scope": "Every release file except this manifest; manifest hash is returned in the release summary",
                "input_sha256": _sha(_json({"ontology": ontology, "sources": sources, "entities": entities,
                    "evidence": evidence, "assertions": assertions, "alignments": alignments, "decisions": decisions,
                    "allow_pending": allow_pending})), "network_calls": False}
    artifacts["manifest.json"] = _json(manifest)
    output_dir.mkdir(parents=True, exist_ok=False)
    # Manifest is committed last; a failed write cannot look like a complete release.
    for name, data in artifacts.items():
        with (output_dir / name).open("xb") as stream:
            stream.write(data)
    manifest_sha = _sha(artifacts["manifest.json"])
    return {"counts": counts, "release_kind": graph["release_kind"],
            "artifact_sha256": {**hashes, "manifest.json": manifest_sha},
            "manifest_sha256": manifest_sha, "review_status": "needs_specialist_review"}
