# Pilot methods in the hosted graph service

The pilot distinguished source-explicit relationships, provisional concept alignments and analytical comparisons. More contextual links alone do not reproduce its curated integration. This update ports those methods and its interactive source-cluster visualization; it does not silently import the historical pilot corpus into the live database.

## Generation and reproduction

`graph_generation_profiles.prepare_generation_config` adds versioned `source-relationships/1` reading guidance to new assisted runs' ontology configuration. The existing extraction prompt includes that configuration, and its exact hash is pinned. Ontology semantics, output schema and quotation validation remain intact. Existing run configurations and every pinned pipeline file stay unchanged.

Generation inventories source-explicit actors, activities, tools, materials, places, periods and stages, using only declared types. Where a passage supports a permitted relationship between independently supported endpoints, it should return a typed entity-to-entity assertion. Measurements, counts and unsupported referents remain literal. No density target is imposed. Co-occurrence, object classification, museum location and broad taxonomic names cannot establish tool use, production place, identity or a more specific taxon.

New create/from-extraction requests, explicit retries and the preparation worker use the profile for assisted extraction. `POST /research-runs/{run_id}/improve` with `{"max_auto_batches":4}` creates an idempotent derived draft and queues it. The API allows 1–20 batches; the interface starts with four. Continue processing for remaining passages. Parent graphs and review histories are preserved.

## Source integration

`graph_inferences.infer_connections` returns separate `supported_relationships` and `inferences` arrays. Recovery requires an existing literal assertion's entire value to uniquely match a permitted same-source concept grounded in the same quoted passage. It retains the original statement through `derived_from_assertion_id`, its evidence, scope and polarity. This is one claim in traversable form, not independent evidence or scientific approval. Descriptive prose is not mined into new facts.

Uncertain proposals cover typed concept/name correspondences, exact quoted definition-name bridges, shared reported paths and shared quotation context. Matching names cannot align physical objects, people or events; place/time matches remain name-context suggestions. Source-family and root-seed metadata are retained where recorded. Source independence, physical identity and historical transmission are not inferred. Rejections, incompatible contexts, duplicate pairs and explicit output limits remain enforced and reported.

Combined snapshots use `graph-network/2`. `POST /graph-networks/{network_id}/refresh` creates a current successor while retaining its predecessor. Selected refresh keeps its selected run IDs; choose newer derived drafts explicitly to change that selection. Library refresh prefers completed graphs, accepted-passage coverage, partially validated coverage, reported connections and concept count. Missing files and unfinished passages remain visible.

## Visualization and structural measures

The canvas supports whole-network, one-source and one/two-hop neighborhood views; pan, zoom, fit, PNG export and keyboard navigation. Optional source attribution and exact scope references remain separate from semantic relationships. Typed colors, negative/uncertain polarity and distinct reported/comparison/inference layers preserve interpretation. The side panel retains original quotations, page locators, full records and source navigation.

Parallel visual triples can aggregate while all original claims remain inspectable. Limits of 5,000 concepts and 12,000 visual connections are explicit. Positions are deterministic display geometry, not geography, similarity or historical movement.

`graph_quality.connectivity_report` counts exact subject/object endpoints for reported relationships and the combined analytical layer separately. Literal descriptions and scope IDs cannot inflate connectivity. Counts include negative and uncertain recorded links and do not measure truth, extraction completeness, corroboration or source independence.

Focused checks:

```sh
.venv/bin/python -m unittest test.test_graph_generation_profiles test.test_graph_inferences test.test_graph_quality test.test_graph_networks test.test_research_runs -q
```

Build a fresh network after additional document processing. Frozen snapshots do not mutate automatically. No dependencies, schema, authentication, environment files or deployment configuration changed.
