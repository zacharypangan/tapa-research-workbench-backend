# Semantic Atlas edge rules

Policy version: `2026-06-19`

The machine-readable source of truth is `app/repository/semantic_rules.py`. Researchers can retrieve the same registry from:

```text
GET /api/v1/repository/graph/semantic/rules
```

An individual accepted or reviewable relation can be audited with:

```text
GET /api/v1/repository/graph/semantic/relations/{relation_id}/explain
```

## Purpose

The Semantic Atlas is a research navigation and evidence-audit layer over repository materials. It projects catalogued metadata, human observations, source-linked mentions, and reviewed analytical relations into a smaller entity/relation graph.

The graph is not a claim that every displayed edge is an objective fact. Each edge has:

- a rule ID;
- an extraction method;
- a confidence value;
- an assertion/review status;
- source-linked evidence;
- an explicit statement of what the edge does and does not mean.

For reproducible analysis, record the rule policy version, graph build date, relation ID, rule ID, review status, and cited evidence locators.

## Semantic graph versus evidence index

The repository has two related but distinct graph layers.

### Semantic graph

The Semantic Atlas is built from the `kg_*` tables:

- `kg_entities`: canonical material, collection, source type, concept, place, time, and agent nodes.
- `kg_mentions`: source-linked mentions of entities.
- `kg_relations`: aggregated semantic assertions shown or available to the atlas.
- `kg_relation_evidence`: evidence units supporting each semantic relation.
- `kg_candidate_relations`: weak analytical suggestions kept outside accepted relations.
- `kg_candidate_evidence`: evidence units supporting candidates.
- `kg_entity_stats`: derived mention, document, collection, degree, and importance statistics.
- `kg_place_resolution` / `kg_time_resolution`: resolution and validation queues.
- `kg_resolution_evidence`: evidence for place/time resolution decisions.
- `kg_review_decisions`: durable human decisions preserved across rebuilds.

### Evidence index

The older `repository_graph_*` tables index document structure, extracted segments, images, observations, and mention edges. They are useful for source traceability and raw graph exploration. They are not the source of accepted semantic meaning. A small amount of legacy review state is migrated into the Semantic Atlas so prior rejections and explicit acceptances are not silently lost.

## Entity types

| Type | Meaning |
| --- | --- |
| `material` | A repository reference or source record. |
| `collection` | A collection label stored in repository metadata. |
| `source_type` | The repository's source classification. |
| `concept` | A normalized topical label grounded in a manual keyword, human observation, or exact reuse of that accepted vocabulary. |
| `place` | A region/place label accepted from metadata or observation, explicitly coordinated, or resolved by review. |
| `time_period` | A publication year or parsed temporal expression that passes validity checks. |
| `agent` | An author/name string projected from metadata; not externally authority-controlled. |

## Relation and candidate rule table

The table below summarizes the implemented rules. Full trigger, evidence, confidence, acceptance, caution, and limitation text is available in the registry endpoint.

| Rule ID | Predicate | Source | Default | Visible | Interpretation |
| --- | --- | --- | --- | --- | --- |
| `structural.collection.metadata` | `belongs_to_collection` | material collection metadata | accepted | yes | Current repository cataloguing membership. |
| `structural.source_type.metadata` | `has_source_type` | material source type metadata | accepted | yes | Current repository source classification. |
| `bibliographic.author.metadata` | `authored_by` | material author metadata | accepted | yes | Metadata attribution to a name string. |
| `semantic.topic.manual_keyword` | `has_topic` | human-edited keyword | accepted | yes | Human-assigned material topic. |
| `semantic.topic.auto_keyword` | `has_topic` | generated keyword | needs review | yes, review-marked | Suggested material topic. |
| `semantic.topic.exact_match` | `has_topic` | exact accepted-vocabulary match in a segment | accepted | yes | Accepted concept label occurs in source text. |
| `semantic.topic.image_ocr` | `has_topic` | exact accepted-vocabulary match in OCR | accepted | yes | Accepted concept label occurs in OCR text. |
| `semantic.topic.image_caption` | `has_topic` | exact accepted-vocabulary match in caption text | accepted | yes | Accepted concept label occurs in caption text. |
| `semantic.topic.aggregated` | `has_topic` | multiple topic methods | accepted if any accepted path | yes | Aggregated material-topic evidence. |
| `semantic.observation_topic.human` | `has_observation_topic` | human observation | accepted | yes | A researcher recorded this concept against the material. |
| `spatial.place.metadata` | `mentions_place` | region metadata | accepted | yes | Material metadata names a region/place. |
| `spatial.place.human_observation` | `mentions_place` | place observation | accepted | yes | A researcher recorded a place against the material. |
| `spatial.place.pattern` | `mentions_place` | pattern plus coordinates/review resolution | needs review | yes, review-marked | A pattern-derived place candidate has a resolvable entity. |
| `spatial.place.aggregated` | `mentions_place` | multiple spatial methods | accepted if any accepted path | yes | Aggregated material-place evidence. |
| `temporal.time.metadata` | `mentions_time` | material year metadata | accepted | yes | Usually the material's publication/catalogued year. |
| `temporal.time.human_observation` | `mentions_time` | resolved date in human observation | accepted | yes | A human observation contains a valid temporal expression. |
| `temporal.time.pattern` | `mentions_time` | valid pattern-derived date | needs review | yes, review-marked | A source passage contains a plausible temporal expression. |
| `temporal.time.aggregated` | `mentions_time` | multiple temporal methods | accepted if any accepted path | yes | Aggregated material-time evidence. |
| `candidate.cooccurrence` | `co_occurs_with` | repeated evidence-unit co-occurrence | needs review | no | Concepts occur in the same evidence units/documents. |
| `semantic.related_to.reviewed_cooccurrence` | `related_to` | accepted co-occurrence candidate | accepted after review | yes | Reviewer-approved, intentionally broad relatedness. |

## Reserved and unimplemented relation types

The registry also documents capabilities that are present in constants/schema or are methodologically relevant but are not currently generated:

- embedding-neighbor candidates;
- relation-pattern candidates;
- LLM-proposed candidates;
- reviewed promotion paths for those candidate methods;
- `same_as`;
- `broader_than`;
- `narrower_than`;
- `near_in_evidence`.

These rules have `implemented=false`, `candidate_only=true`, `visible_by_default=false`, and `default_status=needs_review`. Their presence in the registry is documentation of absence, not evidence that such edges exist.

`near_in_evidence` is especially important: the current system does not define a token, sentence, paragraph, page, or geometric proximity window. It must not be inferred from document-level or evidence-unit co-occurrence.

## How mentions become relations

`kg_mentions` stores the source-level entity evidence. Mention methods currently include:

- `metadata`;
- `keyword`;
- `auto_keyword`;
- `exact_match`;
- `observation`;
- `image_ocr`;
- `image_caption`;
- `pattern_candidate`;
- `llm_candidate` (allowed but not currently generated).

Each mention carries its own confidence and review status. During a build, mentions supporting the same subject-predicate-object triple are grouped.

For an aggregated relation:

- `evidence_count` is the number of unique evidence identities after deduplication;
- `document_count` is the number of distinct `material_id` values among those evidence items;
- confidence is the arithmetic mean of contributing evidence confidences;
- default status is `accepted` when at least one contributing path is accepted, otherwise `needs_review`;
- a stored human relation review decision overrides the generated default;
- contributing rule IDs, source methods, source statuses, and aggregation logic are stored in `properties_json`.

This accepted-if-any policy means an accepted metadata or human assertion can support a relation that also contains review-only pattern evidence. It does not silently convert the individual pattern mention into accepted evidence.

## Confidence policy

Confidence is a property of the extraction/assignment procedure, not a statistical probability that a scholarly proposition is true.

Current fixed or bounded values include:

- collection/source type metadata: `1.0`;
- author metadata: `0.99`;
- manual keyword: `0.98`;
- automatic keyword: `0.72`;
- human observation: `0.99`;
- accepted-vocabulary exact segment match: `0.82`;
- image OCR/caption exact match: `0.76`;
- trusted region metadata: `0.95`;
- publication/material year metadata: `0.98`;
- temporal pattern candidates: approximately `0.55–0.90`, depending on syntax and context;
- place patterns: approximately `0.65–0.72` once an entity can be created;
- co-occurrence: `min(0.90, 0.45 + 0.04 × evidence_count + 0.08 × document_count)`.

Several mentions from one material are not independent replication. Use `document_count` and the evidence list when evaluating breadth.

## Review status policy

Statuses are:

- `accepted`: defensible under the registered rule or explicitly accepted by a reviewer;
- `needs_review`: visible or queued but not established as an accepted semantic assertion;
- `rejected`: retained as a decision but omitted from default graph views.

Direct metadata, human observations, manual keywords, and exact reuse of accepted vocabulary may be accepted by default when the corresponding rule says so.

Pattern, co-occurrence, embedding, and LLM-derived assertions must default to `needs_review`. They may become accepted only through:

1. an explicit relation/candidate review decision; or
2. aggregation with a separate accepted evidence path for the same subject-predicate-object relation.

Candidate acceptance is durable in `kg_review_decisions`. Accepted `co_occurs_with` candidates are promoted to `related_to`, not copied into `kg_relations` as co-occurrence facts.

## Visibility policy

The default Semantic Atlas reads `kg_relations` and excludes rejected relations.

- Accepted and review-needed semantic relations may be returned.
- `kg_candidate_relations` are not rendered as default semantic edges.
- Co-occurrence is always candidate-only until reviewed.
- Generic concepts and low-importance non-bridge concepts may be hidden from overview views without deleting their mentions.
- Unresolved place labels and invalid/ambiguous time values remain in review queues, not the accepted map.

## Place resolution policy

A place relation can be created from:

1. trusted material region metadata;
2. a human observation explicitly typed as `place`;
3. a pattern candidate with valid explicit coordinates; or
4. a saved place-resolution decision.

Unresolved and ambiguous pattern labels are recorded in `kg_place_resolution` with `kg_resolution_evidence`. They do not create accepted place entities or `mentions_place` relations.

The place relation states only that the material is linked to a named place under the rule. It does not specify whether the place is an origin, event location, publication place, comparison, or present-day jurisdiction.

## Time resolution policy

Temporal parsing recognizes explicit years, year ranges, `BP`, `BCE/BC`, and `CE/AD` expressions. It also checks source context for page, figure, table, and plate cues.

Examples such as `100-101` in `pp. 100-101` are stored as `invalid_page_range`. They do not create a `time_period` entity or `mentions_time` relation.

Modern years and explicitly marked eras can be resolved, but segment/image-derived time relations remain `needs_review`. A valid-looking year does not establish what event the year dates.

## Co-occurrence policy

The current co-occurrence unit is a source-linked segment, image, or observation containing more than one accepted concept.

A candidate requires:

- at least three unique evidence units;
- at least two documents;
- non-generic concepts;
- concepts that do not individually occur in more than 30% of the corpus;
- computed confidence of at least `0.55`.

Co-occurrence is symmetric. It does not encode direction, causation, identity, hierarchy, influence, or semantic equivalence. Acceptance promotes the pair to broad `related_to` and preserves the original candidate ID/predicate in provenance.

## Valid interpretation examples

- “Under rule `structural.collection.metadata`, this repository record is catalogued in Collection A.”
- “Under rule `semantic.topic.exact_match`, the accepted concept label ‘barkcloth’ occurs in the cited extracted segment.”
- “This `mentions_time` edge is review-needed pattern evidence for a date expression in page 3; it does not yet establish the date's role.”
- “These concepts produced a hidden co-occurrence candidate in four evidence units across two documents.”
- “This `related_to` edge was promoted only after a reviewer accepted the co-occurrence candidate.”

## Invalid over-interpretation examples

- “Two materials in the same collection have the same historical provenance.”
- “An `authored_by` name string uniquely identifies the historical person.”
- “A `has_topic` edge means the source endorses the concept.”
- “A `mentions_place` edge proves an event happened at that place.”
- “A `mentions_time` edge dates the material object rather than a cited event.”
- “High confidence means a 95% probability the scholarly claim is true.”
- “Co-occurrence proves concepts are equivalent, causally related, or close together in the text.”
- “Repeated evidence snippets from one document are independent corroboration.”

## Limitations and reproducibility notes

- Entity normalization is label-based and does not yet use external authority control.
- Author, place, and concept homonyms may remain conflated until reviewed.
- Exact matches do not perform word-sense disambiguation, stance detection, negation handling, or quotation attribution.
- OCR and caption text can contain extraction errors.
- Pattern-derived places and dates depend on English-oriented cues and heuristics.
- Confidence values are methodological weights, not calibrated probabilities.
- Graph rebuilds preserve explicit review decisions, but source metadata or extracted evidence changes can change evidence counts and aggregate confidence.
- The rules endpoint should be archived with exports used in publications.
- For each reported edge, retain relation ID, rule ID, policy version, status, extraction method, evidence locators, and review notes.
