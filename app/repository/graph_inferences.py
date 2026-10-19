"""Deterministic, evidence-linked connection proposals; no new source facts.

Kept outside the pinned extraction implementation. Original records and approvals
are never rewritten; generated connections are an explicitly separate layer.
"""
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import re
import unicodedata

METHOD_VERSION = "evidence-connections-v2"
MAX_RECOVERY_COMPARISONS = 100000
_STOP = set("a an and article concept document figure introduction material method paper research source study table the this that these those thing things tool type unknown unspecified".split())
_REJECTED = {"reject", "rejected", "exclude", "excluded"}
_CONTEXT = ("language", "place", "time", "scheme", "language_id", "place_id", "time_id", "scheme_id", "typology_scheme_id", "production_case_id", "case_id", "namespace", "taxonomy_namespace", "taxon_rank", "rank")
# A shared name is never enough to align distinguishable objects or events.
_CONCEPT_TYPES = {"Concept", "ActivityType", "ArtifactType", "Material", "MaterialKind", "ToolForm", "PlantTaxon", "Feature", "DesignConcept", "Role", "EnvironmentContext", "TypeCategory", "Language", "LanguageVariety"}
_NAME_TYPES = {"Place", "TimeExpression"}
_SOURCE_REPORTS = {"source_report", "source_reported", "documented_source_report", "secondary_ethnographic_report"}
_RECOVERY_RULES = {
    "has_material": (None, {"Material", "MaterialKind"}),
    "made_of": (None, {"Material", "MaterialKind"}),
    "has_feature": (None, {"Concept", "Feature"}),
    "has_definition": (None, _CONCEPT_TYPES - {"TypeCategory"}),
    "located_in": (None, {"Place"}),
    "occurred_at": ({"Activity"}, {"Place"}),
    "dated_to": (None, {"TimeExpression"}),
    "uses": ({"Person", "Organization", "Activity", "ActivityType", "ReportedStage"}, {"Artifact", "ArtifactType", "ToolInstance", "Material", "MaterialKind", "MaterialBatch", "Concept", "Language"}),
    "produces": ({"Person", "Organization", "Activity", "ActivityType", "ReportedStage"}, {"Artifact", "ArtifactType", "Material", "MaterialKind", "MaterialBatch", "Document", "Concept"}),
    "assigned_type_by_source": (None, {"TypeCategory", "ToolForm", "ArtifactType"}),
    "source_classifies_as_tool_form": ({"ArchaeologicalObject", "ToolInstance", "Artifact"}, {"ToolForm", "ArtifactType"}),
    "category_describes_tool_form": ({"TypeCategory"}, {"ToolForm"}),
    "has_activity": ({"ProductionCase"}, {"ReportedStage", "ActivityType"}),
    "has_activity_type": ({"ReportedStage"}, {"ActivityType"}),
}


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
            if 2 <= len(normalized) <= 200 and normalized not in _STOP and not normalized.isnumeric():
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


def _quoted_name(label, text):
    """Match original spelling with word boundaries; never invent a translation."""
    label, text = _normal(label), _normal(text)
    if not label:
        return False
    left = r"(?<!\w)" if re.match(r"[^\W\d_]", label[0], re.UNICODE) and not re.match(r"[\u3400-\u9fff\u3040-\u30ff]", label[0]) else ""
    right = r"(?!\w)" if re.match(r"[^\W\d_]", label[-1], re.UNICODE) and not re.match(r"[\u3400-\u9fff\u3040-\u30ff]", label[-1]) else ""
    return bool(re.search(left + re.escape(label) + right, text))


def _overlapping_evidence(first_refs, second_refs, evidence):
    for first_ref, second_ref in itertools.product(first_refs, second_refs):
        if first_ref == second_ref:
            return True
        first, second = evidence[first_ref], evidence[second_ref]
        if first.get("source_id") != second.get("source_id") or not first.get("unit_id") or first.get("unit_id") != second.get("unit_id"):
            continue
        first_bounds, second_bounds = _bounds(first), _bounds(second)
        if first_bounds and second_bounds and max(first_bounds[0], second_bounds[0]) < min(first_bounds[1], second_bounds[1]):
            return True
    return False


def _provenance(records, evidence_ids, entities, evidence, sources):
    records = [*records, *(evidence[ref] for ref in evidence_ids)]
    source_ids = sorted({row.get("source_id") for row in records if row.get("source_id") in sources})
    source_rows = [sources[key] for key in source_ids]
    records += source_rows

    def ids(field):
        return sorted({value for row in records for value in (row.get(field, []) if isinstance(row.get(field, []), list) else []) if isinstance(value, str) and value})

    families = sorted({value for row in records for value in (row.get("source_family_id"), _scope(row).get("source_family_id")) if isinstance(value, str) and value})
    origins = {key: sources[key].get("origin") if isinstance(sources[key].get("origin"), str) else "unspecified" for key in source_ids}
    return {"source_ids": source_ids, "source_family_ids": families,
            "source_independence": "same_declared_family" if len(source_ids) > 1 and len(families) == 1 and all(row.get("source_family_id") == families[0] for row in source_rows) else "not_established",
            "source_origins": origins, "cross_source": len(source_ids) > 1,
            "seed_ids": ids("seed_ids"), "root_seed_ids": ids("root_seed_ids"),
            "primary_entity_ids": sorted({row["entity_id"] for row in records if row.get("entity_id") in entities and sources[entities[row["entity_id"]]["source_id"]].get("origin") == "primary"}),
            "connects_primary_to_augmented": "primary" in origins.values() and any(
                row.get("origin") == "web" or row.get("source_kind") in {"web", "html", "web_page", "web_snapshot"}
                for row in source_rows)}


def infer_connections(graph, *, max_connections=2000, max_per_entity=12, max_supported_relationships=1000):
    """Recover exact literal links and propose scoped correspondences, without merging IDs.

    ``supported_relationships`` restates retained source assertions in traversable
    form. ``inferences`` remains a separate, uncertain layer; neither is approved.
    """
    if type(max_connections) is not int or not 1 <= max_connections <= 5000:
        raise ValueError("max_connections must be an integer from 1 to 5000")
    if type(max_per_entity) is not int or not 1 <= max_per_entity <= 50:
        raise ValueError("max_per_entity must be an integer from 1 to 50")
    if type(max_supported_relationships) is not int or not 1 <= max_supported_relationships <= 5000:
        raise ValueError("max_supported_relationships must be an integer from 1 to 5000")
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
        if statement.get("source_id") not in sources or any(evidence[ref]["source_id"] != statement["source_id"] for ref in statement["evidence_ids"]):
            raise ValueError("Statement evidence belongs to another source")

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
    def source_key(statement, object_id):
        return (statement["subject_id"], statement["predicate"], object_id, statement.get("polarity", "uncertain"), json.dumps(statement.get("scope", {}), sort_keys=True, ensure_ascii=False))

    existing_source_keys = {source_key(statement, statement["object_id"]) for statement in [*assertions.values(), *graph.get("supported_relationships", [])] if statement.get("object_id")}
    for statement in [*graph.get("alignments", []), *graph.get("inferences", []), *graph.get("supported_relationships", [])]:
        if statement.get("subject_id") in entities and statement.get("object_id") in entities:
            existing.add(tuple(sorted((statement["subject_id"], statement["object_id"]))))
    connections, recovered, degrees, counters, seen = [], [], Counter(), Counter(), set()
    truncated = False
    recovery_truncated = False
    recovery_considered = 0
    considered = 0
    pair_budget = 100000

    def tick():
        nonlocal considered, truncated
        if considered >= pair_budget:
            truncated = True
            return False
        considered += 1
        return True

    # Reify only an exact, source-original literal value that uniquely resolves
    # to a type-appropriate entity grounded in the very same quoted passage.
    # Descriptive prose, generated English labels and mere word mentions do not
    # become uses/material/location assertions through this adapter.
    literal_candidates = defaultdict(list)
    for entity_id, entity in sorted(eligible.items()):
        label = entity.get("label_original")
        if isinstance(label, str) and 2 <= len(_normal(label)) <= 200:
            literal_candidates[(entity["source_id"], _normal(label))].append(entity_id)
    for statement_id, statement in sorted(assertions.items()):
        if statement.get("object_id") or not isinstance(statement.get("value"), str) or excluded(statement, statement_id):
            continue
        rule = _RECOVERY_RULES.get(statement.get("predicate"))
        subject_id = statement["subject_id"]
        if not rule or subject_id not in eligible or statement.get("epistemic_kind") not in _SOURCE_REPORTS:
            continue
        subject = eligible[subject_id]
        source_id = statement.get("source_id")
        refs = statement["evidence_ids"]
        if source_id != subject["source_id"] or any(evidence[ref]["source_id"] != source_id for ref in refs):
            continue
        subject_types, object_types = rule
        if subject_types and subject["entity_type"] not in subject_types:
            continue
        matches = []
        for object_id in literal_candidates.get((source_id, _normal(statement["value"])), []):
            if recovery_considered >= MAX_RECOVERY_COMPARISONS:
                recovery_truncated = True
                break
            recovery_considered += 1
            target = eligible[object_id]
            if object_id == subject_id or target["entity_type"] not in object_types or not _compatible(subject, target, statement):
                continue
            quoted = [ref for ref in refs if _quoted_name(target["label_original"], evidence[ref].get("text", ""))]
            if quoted and _overlapping_evidence(quoted, target["evidence_ids"], evidence):
                matches.append(object_id)
        if recovery_truncated:
            break  # A partial candidate scan must not masquerade as unique resolution.
        if len(matches) != 1:
            counters["ambiguous_literal_target" if matches else "unresolved_literal_target"] += 1
            continue
        object_id = matches[0]
        if source_key(statement, object_id) in existing_source_keys:
            continue
        if len(recovered) >= max_supported_relationships:
            recovery_truncated = True
            break
        target = eligible[object_id]
        recovered_refs = sorted(set(refs + subject["evidence_ids"] + target["evidence_ids"]))
        seed = [METHOD_VERSION, "literal_relation_recovery", statement_id, object_id, recovered_refs]
        identity = hashlib.sha256(json.dumps(seed, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:24]
        recovered.append({"assertion_id": "REC_" + identity, "source_id": source_id,
            "subject_id": subject_id, "object_id": object_id, "predicate": statement["predicate"],
            "polarity": statement.get("polarity", "uncertain"), "epistemic_kind": "source_report",
            "origin_epistemic_kind": statement["epistemic_kind"],
            "scope": copy.deepcopy(statement.get("scope", {})), "evidence_ids": recovered_refs,
            "premise_ids": [statement_id], "derived_from_assertion_id": statement_id,
            "origin": "literal_relation_recovery", "rule": "exact_literal_entity_recovery",
            "literal_value": statement["value"], "review_status": "needs_review",
            "scientific_status": "source_claim_reexpression", "fact_transfer": False,
            "method_version": METHOD_VERSION,
            "rationale": "The original source assertion's entire literal value matches one type-appropriate, source-local concept whose original label is explicitly quoted in the same passage. This traversable representation preserves the original claim, polarity and scope; it is not independent evidence or specialist approval.",
            **_provenance([subject, target, statement], recovered_refs, entities, evidence, sources)})
        existing.add(tuple(sorted((subject_id, object_id))))
        existing_source_keys.add(source_key(statement, object_id))

    def propose(left_id, right_id, rule, predicate, refs, rationale, *, premise_ids=(), via=(), metadata=None, layer="context_association"):
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
                "layer": layer,
                "review_status": "needs_review", "scientific_status": "unverified_proposal",
                "fact_transfer": False, "method_version": METHOD_VERSION,
                **_provenance([eligible[left_id], eligible[right_id], *(eligible[key] for key in via), *(assertions[key] for key in premise_ids)], refs, entities, evidence, sources),
                **(metadata or {})}
        connections.append(item)
        degrees.update(pair)
        counters[rule] += 1
        seen.add(pair)

    # Cross-record comparison, not entity resolution: aliases retain their origin.
    groups, label_fields = defaultdict(list), {}
    for key, entity in sorted(eligible.items()):
        if entity.get("entity_type") not in _CONCEPT_TYPES | _NAME_TYPES or entity.get("is_instance") is True:
            counters["nonalignable_entities"] += 1
            continue
        fields = _labels(entity)
        label_fields[key] = fields
        for value in fields:
            groups[(entity.get("entity_type"), value)].append(key)
    for (_, value), group in sorted(groups.items(), key=lambda row: str(row[0])):
        for left, right in itertools.combinations(sorted(group), 2):
            if not tick():
                break
            translated = not (label_fields[left][value] - {"label_en"} and label_fields[right][value] - {"label_en"})
            name_only = eligible[left].get("entity_type") in _NAME_TYPES
            propose(left, right, "typed_label_match", "has_related_concept_context" if name_only else "aligns_reported_concept_with",
                    eligible[left]["evidence_ids"] + eligible[right]["evidence_ids"],
                    ("The reported names and recorded types match; this is a name correspondence, not proof of the same geographic extent or temporal interval. " if name_only else "The labels and recorded generic concept types match. ") +
                    "These remain separate source accounts; identity, chronology and historical transmission are not established. " +
                    ("The match uses a proposed English translation that also needs review." if translated else "Missing context does not establish equivalence."),
                    metadata={"matched_label": value, "translation_involved": translated,
                              "mapping_kind": "reported_name_correspondence" if name_only else "typed_concept_correspondence"},
                    layer="concept_alignment")
            if considered >= pair_budget:
                break
        if considered >= pair_budget:
            break

    # A retained short definition can provide a source-attributed bridge between
    # otherwise different names, without treating the definition as an identity
    # assertion or transferring the properties of the described category.
    original_names = defaultdict(list)
    for key, entity in sorted(eligible.items()):
        if entity.get("entity_type") in _CONCEPT_TYPES and entity.get("is_instance") is not True and isinstance(entity.get("label_original"), str):
            value = _normal(entity["label_original"])
            if 2 <= len(value) <= 120 and value not in _STOP and not value.isnumeric():
                original_names[value].append(key)
    for key, statement in sorted(assertions.items()):
        value = statement.get("value")
        left = statement["subject_id"]
        if statement.get("predicate") not in {"has_definition", "attested_term_for"} or not isinstance(value, str) or left not in eligible or excluded(statement, key):
            continue
        if eligible[left].get("entity_type") not in _CONCEPT_TYPES or eligible[left].get("is_instance") is True or statement.get("polarity", "positive") != "positive" or statement.get("epistemic_kind") not in _SOURCE_REPORTS:
            continue
        if not any(evidence[ref].get("source_id") == eligible[left]["source_id"] and _quoted_name(value, evidence[ref].get("text", "")) for ref in statement["evidence_ids"]):
            continue
        for right in original_names.get(_normal(value), []):
            if eligible[right]["source_id"] == eligible[left]["source_id"]:
                continue
            if not tick():
                break
            if not _compatible(eligible[left], eligible[right], statement):
                continue
            propose(left, right, "source_definition_name_match", "has_related_concept_context",
                    statement["evidence_ids"] + eligible[left]["evidence_ids"] + eligible[right]["evidence_ids"],
                    "A retained source assertion defines one term with an exact quoted name also reported by the other source. This is a source-qualified term correspondence, not accepted equivalence, identity, independent corroboration or transfer of properties between the accounts.",
                    premise_ids=(key,), metadata={"matched_label": _normal(value), "translation_involved": False,
                                                  "mapping_kind": "definition_to_reported_name"}, layer="concept_alignment")
        if considered >= pair_budget:
            break

    # Shared endpoint paths describe recorded structure only, not factual closure.
    neighbors = defaultdict(list)
    for key, statement in sorted(assertions.items()):
        left, right = statement["subject_id"], statement.get("object_id")
        if not right or left not in eligible or right not in eligible or excluded(statement, key):
            continue
        if statement.get("polarity", "positive") != "positive" or statement.get("epistemic_kind") not in (None, *_SOURCE_REPORTS):
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
    return {"inferences": connections, "supported_relationships": recovered, "report": {"method_version": METHOD_VERSION,
            "generated": len(connections), "by_rule": {key: counters[key] for key in ("typed_label_match", "source_definition_name_match", "shared_reported_endpoint", "shared_quotation_context")},
            "supported_relationships": len(recovered), "supported_relationship_rule": "exact_literal_entity_recovery",
            "supported_relationships_truncated": recovery_truncated,
            "recovery_considered_candidates": recovery_considered, "recovery_candidate_budget": MAX_RECOVERY_COMPARISONS,
            "ambiguous_literal_targets": counters["ambiguous_literal_target"], "unresolved_literal_targets": counters["unresolved_literal_target"],
            "considered_pairs": min(considered, pair_budget), "skipped_incompatible_context": counters["incompatible_context"],
            "excluded_entities": len(entities) - len(eligible), "nonalignable_entities": counters["nonalignable_entities"], "truncated": truncated,
            "max_connections": max_connections, "max_per_entity": max_per_entity,
            "max_supported_relationships": max_supported_relationships,
            "review_required": True, "facts_inferred": False,
            "note": "Deterministic, bounded source-claim reexpressions, typed concept/name correspondences and contextual navigation proposals from saved evidence. Source relationships retain their original premise and polarity; proposals never transfer facts or merge entities. Source-family independence is not established by a label match."}}
