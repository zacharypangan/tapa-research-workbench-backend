"""Deterministic, evidence-linked connection proposals; no new source facts.

Kept outside the pinned extraction implementation. Original records and approvals
are never rewritten; generated connections are an explicitly separate layer.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import re
import unicodedata

METHOD_VERSION = "evidence-connections-v1"
_STOP = set("a an and article concept document figure introduction material method paper research source study table the this that these those thing things tool type unknown unspecified".split())
_REJECTED = {"reject", "rejected", "exclude", "excluded"}
_CONTEXT = ("language", "place", "time", "scheme", "language_id", "place_id", "time_id", "scheme_id", "typology_scheme_id", "production_case_id", "case_id", "namespace", "taxonomy_namespace", "taxon_rank", "rank")


def _normal(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", value)
    return re.sub(r"\s+", " ", re.sub(r"[‐‑–—]", "-", value))


def _labels(entity):
    fields = defaultdict(set)
    for key in ("label_original", "label_en", "aliases"):
        values = entity.get(key, []) if key == "aliases" else [entity.get(key)]
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str):
                continue
            normalized = _normal(value)
            if len(normalized) >= 2 and normalized not in _STOP:
                fields[normalized].add(key)
    return fields


def _scope(record):
    scope = dict(record.get("scope") or {})
    for key in _CONTEXT:
        if record.get(key) is not None:
            scope.setdefault(key, record[key])
    return scope


def _compatible(*records):
    scopes = [_scope(record) for record in records]
    for key in _CONTEXT:
        values = {json.dumps(scope[key], sort_keys=True, ensure_ascii=False) for scope in scopes if scope.get(key) not in (None, "")}
        if len(values) > 1:
            return False
    # Classification codes only compare inside an explicitly shared scheme.
    if any(record.get("entity_type") == "TypeCategory" for record in records):
        schemes = [scope.get("scheme_id", scope.get("typology_scheme_id")) for scope in scopes]
        if not all(schemes) or len(set(map(str, schemes))) != 1:
            return False
    return True


def _index(graph, field, key):
    rows = graph.get(field, [])
    if not isinstance(rows, list):
        raise ValueError(field + " must be a list")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key] or row[key] in result:
            raise ValueError("Invalid or duplicate " + key)
        result[row[key]] = row
    return result


def _bounds(evidence):
    locator = evidence.get("character_locator") or {}
    spans = locator.get("spans", []) if isinstance(locator, dict) else []
    spans = [span for span in spans if isinstance(span, dict) and type(span.get("start")) is int and type(span.get("end")) is int and 0 <= span["start"] < span["end"]]
    return (min(span["start"] for span in spans), max(span["end"] for span in spans)) if spans else None


def infer_connections(graph, *, max_connections=2000, max_per_entity=12):
    """Return unreviewed hypotheses with premises; never mutate graph or merge IDs."""
    if type(max_connections) is not int or not 1 <= max_connections <= 5000:
        raise ValueError("max_connections must be an integer from 1 to 5000")
    if type(max_per_entity) is not int or not 1 <= max_per_entity <= 50:
        raise ValueError("max_per_entity must be an integer from 1 to 50")
    entities = _index(graph, "entities", "entity_id")
    evidence = _index(graph, "evidence", "evidence_id")
    sources = _index(graph, "sources", "source_id")
    assertions = _index(graph, "assertions", "assertion_id")
    for item in evidence.values():
        if item.get("source_id") not in sources:
            raise ValueError("Unknown evidence source")
    for entity in entities.values():
        refs = entity.get("evidence_ids", [])
        if entity.get("source_id") not in sources or not refs or any(ref not in evidence for ref in refs):
            raise ValueError("Unknown entity source or evidence")
        if any(evidence[ref]["source_id"] != entity["source_id"] for ref in refs):
            raise ValueError("Entity evidence belongs to another source")
    for statement in assertions.values():
        if statement.get("subject_id") not in entities or (statement.get("object_id") is not None and statement["object_id"] not in entities):
            raise ValueError("Unknown statement endpoint")
        if not statement.get("evidence_ids") or any(ref not in evidence for ref in statement["evidence_ids"]):
            raise ValueError("Unknown statement evidence")

    def decision_time(decision):
        try:
            value = datetime.fromisoformat(str(decision.get("at", "")).replace("Z", "+00:00"))
            return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).timestamp()
        except (ValueError, TypeError, OverflowError):
            return float("-inf")

    latest = {}
    for decision in [*graph.get("origin_decisions", []), *graph.get("decisions", [])]:
        key = decision.get("candidate_id")
        if key:
            previous = latest.get(key)
            if previous is None or decision_time(decision) >= decision_time(previous):
                latest[key] = decision

    def excluded(record, key):
        return (latest.get(key, {}).get("decision") in _REJECTED or
                any(record.get(field) in _REJECTED for field in ("review_status", "status", "inclusion_decision", "origin_review_decision")))

    eligible = {key: entity for key, entity in entities.items() if not excluded(entity, key)}
    existing = {tuple(sorted((statement["subject_id"], statement["object_id"]))) for statement in assertions.values() if statement.get("object_id")}
    for statement in [*graph.get("alignments", []), *graph.get("inferences", [])]:
        if statement.get("subject_id") in entities and statement.get("object_id") in entities:
            existing.add(tuple(sorted((statement["subject_id"], statement["object_id"]))))
    connections, degrees, counters, seen = [], Counter(), Counter(), set()
    truncated = False
    considered = 0
    pair_budget = 100000

    def tick():
        nonlocal considered, truncated
        if considered >= pair_budget:
            truncated = True
            return False
        considered += 1
        return True

    def propose(left_id, right_id, rule, predicate, refs, rationale, *, premise_ids=(), via=(), metadata=None):
        nonlocal truncated
        pair = tuple(sorted((left_id, right_id)))
        if left_id == right_id or left_id not in eligible or right_id not in eligible or pair in seen or pair in existing:
            return
        if not _compatible(eligible[left_id], eligible[right_id]):
            counters["incompatible_context"] += 1
            return
        if len(connections) >= max_connections or any(degrees[key] >= max_per_entity for key in pair):
            truncated = True
            return
        refs = sorted(set(refs))
        if not refs or any(ref not in evidence for ref in refs):
            return
        seed = [METHOD_VERSION, pair, rule, sorted(premise_ids), sorted(via), refs]
        identity = hashlib.sha256(json.dumps(seed, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:24]
        item = {"inference_id": "INF_" + identity, "subject_id": pair[0], "object_id": pair[1],
                "predicate": predicate, "epistemic_kind": "inference", "polarity": "uncertain", "scope": {},
                "evidence_ids": refs, "rule": rule, "rationale": rationale,
                "premise_ids": sorted(set(premise_ids)), "via_entity_ids": sorted(set(via)),
                "review_status": "needs_review", "scientific_status": "unverified_proposal",
                "fact_transfer": False, "method_version": METHOD_VERSION, **(metadata or {})}
        connections.append(item)
        degrees.update(pair)
        counters[rule] += 1
        seen.add(pair)

    # Cross-record comparison, not entity resolution: aliases retain their origin.
    groups, label_fields = defaultdict(list), {}
    for key, entity in sorted(eligible.items()):
        fields = _labels(entity)
        label_fields[key] = fields
        for value in fields:
            groups[(entity.get("entity_type"), value)].append(key)
    for (_, value), group in sorted(groups.items(), key=lambda row: str(row[0])):
        for left, right in itertools.combinations(sorted(group), 2):
            if not tick():
                break
            translated = not (label_fields[left][value] - {"label_en"} and label_fields[right][value] - {"label_en"})
            propose(left, right, "typed_label_match", "possibly_same_concept",
                    eligible[left]["evidence_ids"] + eligible[right]["evidence_ids"],
                    "The labels and recorded concept types match. These remain separate source accounts; identity, chronology and historical transmission are not established. " +
                    ("The match uses a proposed English translation that also needs review." if translated else "Missing context does not establish equivalence."),
                    metadata={"matched_label": value, "translation_involved": translated})
            if considered >= pair_budget:
                break
        if considered >= pair_budget:
            break

    # Shared endpoint paths describe recorded structure only, not factual closure.
    neighbors = defaultdict(list)
    for key, statement in sorted(assertions.items()):
        left, right = statement["subject_id"], statement.get("object_id")
        if not right or left not in eligible or right not in eligible or excluded(statement, key):
            continue
        if statement.get("polarity", "positive") != "positive" or statement.get("epistemic_kind") not in (None, "source_report", "source_reported", "documented_source_report"):
            continue
        neighbors[left].append((right, key))
        neighbors[right].append((left, key))
    for center, paths in sorted(neighbors.items()):
        for (left, first), (right, second) in itertools.combinations(sorted(set(paths)), 2):
            if not tick():
                break
            if not _compatible(assertions[first], assertions[second], eligible[left], eligible[right]):
                continue
            propose(left, right, "shared_reported_endpoint", "shares_reported_connection",
                    assertions[first]["evidence_ids"] + assertions[second]["evidence_ids"] + eligible[left]["evidence_ids"] + eligible[right]["evidence_ids"],
                    "Both concepts have a reported connection to the same recorded concept. The two cited premises support this navigation path; they do not assert a direct, causal, temporal or historical relationship between these concepts.",
                    premise_ids=(first, second), via=(center,))
            if considered >= pair_budget:
                break
        if considered >= pair_budget:
            break

    # First prefer an identical quoted passage; then nearby quotes in one unit.
    contexts = defaultdict(lambda: defaultdict(list))
    for key, entity in sorted(eligible.items()):
        for ref in sorted(set(entity["evidence_ids"])):
            item = evidence[ref]
            contexts[("quote", ref)][key].append(ref)
            if item.get("unit_id") and _bounds(item):
                contexts[("unit", item["source_id"], item["unit_id"])][key].append(ref)
    for context, members in sorted(contexts.items()):
        for left, right in itertools.combinations(sorted(members), 2):
            if not tick():
                break
            choices = []
            for first_ref, second_ref in itertools.product(members[left], members[right]):
                if context[0] == "unit":
                    first, second = _bounds(evidence[first_ref]), _bounds(evidence[second_ref])
                    distance = max(first[0] - second[1], second[0] - first[1], 0)
                    if distance > 400:
                        continue
                else:
                    distance = 0
                choices.append((distance, first_ref, second_ref))
            if not choices:
                continue
            _, first_ref, second_ref = min(choices)
            refs = (first_ref, second_ref)
            propose(left, right, "shared_quotation_context", "shares_quoted_context", refs,
                    "The concepts occur in the same quoted passage or in nearby quotations within one source text unit. This is a contextual association for investigation, not a reported relationship or evidence of cultural or historical transmission.")
            if considered >= pair_budget:
                break
        if considered >= pair_budget:
            break
    return {"inferences": connections, "report": {"method_version": METHOD_VERSION,
            "generated": len(connections), "by_rule": {key: counters[key] for key in ("typed_label_match", "shared_reported_endpoint", "shared_quotation_context")},
            "considered_pairs": min(considered, pair_budget), "skipped_incompatible_context": counters["incompatible_context"],
            "excluded_entities": len(entities) - len(eligible), "truncated": truncated,
            "max_connections": max_connections, "max_per_entity": max_per_entity,
            "review_required": True, "facts_inferred": False,
            "note": "Deterministic, bounded navigation proposals from saved evidence. No concepts are merged and no new historical, causal, geographic or chronological facts are asserted."}}
