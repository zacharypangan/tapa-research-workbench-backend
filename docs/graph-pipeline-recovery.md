# Reliable document graphs and bounded augmentation

Implemented 19 October 2026. This is an implementation contract; candidate validation and operational stopping are not scientific acceptance or completeness.

## Reproducible processing

1. Registration creates immutable graph input snapshots, source locators and SHA-256 hashes. The run pins the pipeline implementation and ontology. Graph ingestion uses the shared `document_representation.extract_document` native/OCR representation also used by repository extraction. TXT/Markdown spans, JSON leaf pointers and DOCX XML/table locators retain exact text. PDF OCR units retain page, language, method and confidence. Generated visual descriptions never become quotations from a paper.
2. Batches cover all readable units, splitting long units with explicit parent offsets. The model receives deterministic sentence/500-character span IDs and the ontology. Selecting a registered `unit_id` plus `span_id`, or explicit Unicode character offsets, derives the quotation directly from immutable text. Legacy exact quotations remain supported. Unknown/out-of-range selections, unsupported labels, dangling references and predicate/scope violations are rejected by the same validator.
3. Ollama Cloud receives a best-effort JSON prompt without the unsupported `format` schema ([Ollama structured-output documentation](https://docs.ollama.com/capabilities/structured-outputs)). Local runtimes receive the schema request; neither configuration is represented as verified enforcement. A maximum of three requests per batch handles malformed JSON, grounding errors, 429 and transient failures, with 1/2-second backoff, 90-second request timeout and an 8,192-output-token ceiling per request. Authentication failures are not retried. Attempts and their validation results are saved separately. An invalid final batch remains blocked; independently valid candidates are preserved in `validated_candidates/`, never silently published.
4. Primary entities generate questions and seeds. Web acquisition remains tied to those seed IDs. Optional Crossref discovery makes at most five sequential public scholarly search requests (at most five results each, at least 0.5 seconds apart), saves the discovery responses/hashes, ranks title overlap against original/translated seed labels and topic, deduplicates candidate URLs, and obeys the run's 1–25-page cap. Registry metadata is URL discovery only; it does not supply scientific graph claims. No external credentials are needed. [Crossref API documentation](https://github.com/CrossRef/rest-api-doc).
5. Registered URLs are collected with the existing public-address/robots/static-snapshot fidelity checks. Collection occurs in checkpointed windows; extraction and validation finish before the next window. Every successful readable page contributes validated novel candidate counts. Defaults are five pages per window, at least ten successful pages, two consecutive windows below one novel candidate per page, or at least 90% duplicates across those windows. The explicit page cap or exhausted frontier can stop earlier. These are configurable operational thresholds. Failed/empty fetches do not count as successful pages. Replay reuses saved discovery, snapshots and processed batches.
6. `augmentation_relevance.json` records each added entity's seed IDs, original evidence, and whether it has a two-sided alignment proposal, a source-reported relationship to a seed match, or only page-level seed context. Page-level context does not establish relevance; all remain reviewable. Alignment proposes comparison and never transfers facts or merges identities.
7. Search returns an additional `graph_results` list with original quotations, neighbours, run ID, exact source locator, draft/released state, inclusion status and outstanding scientific review. This is a bounded small-corpus checkpoint scan, separate from passage scores and embeddings. Material filters apply to individual source entities; web entities require explicit lineage to that material's seeds.

## Public API

The `source-local-extraction-v5-addressable-evidence` prompt uses short request-local unit addresses with an exact audited mapping to immutable IDs. An original label must appear in its own selected evidence, including original OCR spacing. Repair feedback includes bounded matching/selected source spans and endpoint-unit hints for relationships. These are selection hints, not accepted facts or automatic corrections. Original responses, mappings, rejection reasons and attempts remain in the audit; the validator is unchanged.

Derived retries may reuse a parent's completed primary response only when document descriptors, ontology, batch and units are identical and current validation produces identical entities, assertions and evidence. Original provider metadata remain with explicit reuse provenance, parent response/checkpoint hashes and current implementation hashes. Any mismatch/rejection uses normal extraction. Rejected parent batches and web augmentation do not qualify.

- `POST /api/v1/research-runs` accepts the existing config plus:
  ```json
  {"augmentation":{"enabled":true,"discovery":"crossref","max_pages":15,"window_size":5,"min_successful_pages":10,"low_yield_windows":2,"min_avg_novel_entities":1.0,"duplicate_ceiling":0.9}}
  ```
  Manual discovery remains the default. Source-only runs use `enabled:false`.
- `POST /api/v1/research-runs/{id}/discover` records a discovery request and queues the worker for a run awaiting web sources. A saved unsuccessful discovery is not queried repeatedly on reload; researchers can register URLs explicitly.
- `POST /api/v1/research-runs/{id}/retry` with `{}` creates an idempotent **new derived run** from the original immutable document bytes, using the current implementation. The response identifies the new `run_id` and `parent_run_id`. Optional `config`, `title` and `Idempotency-Key` are supported. Original annotations, decisions, releases, failures and snapshots remain unchanged. A supplied `config` is a complete replacement configuration after defaults, not a partial patch of the original.
- `POST /api/v1/research-runs/{id}/resume` still resumes unchanged pinned implementation/checkpoints. After an implementation change use `/retry`; editing old implementation hashes is prohibited.
- `GET /api/v1/research-runs/ontologies/{general|tapa}` returns versioned templates. General research covers people, organizations, places, times, activities, artifacts, materials, languages, documents and concepts. The API's existing tapa default stays compatible; callers explicitly pin their chosen complete ontology in config.
- `/capabilities` reports best-effort/schema-requested extraction, the processing version, independent validation, retry limits, evidence selection methods, ontology templates and available discovery. Graph scheduler restart resets its event for the new asyncio loop.

## Audit files

`ingest.json`, per-document `transcription.json`, `*_batches.json`, `provider_attempts/{batch}/attempt-NN.json`, `rejected_batches/`, `validated_candidates/`, `processed/`, `discovery/crossref.json`, `discovery_request.json`, `web_rounds/`, `collections/round-NNN/`, `augmentation_progress.json`, `augmentation_metrics.json`, `augmentation_relevance.json`, `graph_input.json`, immutable reviewed `releases/`.

## Verification

```bash
.venv/bin/python -m unittest \
  test.test_knowledge_pipeline_processing test.test_knowledge_pipeline_service \
  test.test_knowledge_pipeline_adapters test.test_knowledge_pipeline_contracts \
  test.test_knowledge_pipeline_discovery test.test_knowledge_pipeline_graph \
  test.test_knowledge_pipeline_runtime test.test_research_runs
```

Fixtures cover strict malformed/unsupported outputs, cloud/local capability requests, exact Unicode span selection and invalid offsets, bounded provider repair/transient/auth handling and attempt audit, OCR-to-graph provenance, immutable explicit retry/idempotency, seed-specific discovery snapshots and unsafe/unrelated URL rejection, rolling yield stop before the hard cap, cache replay without collection/provider calls, and per-material graph search isolation.

Remaining limits: Crossref covers scholarly DOI metadata rather than general web/social media; title ranking does not guarantee relevance or accessible full text. Static collection may encounter unavailable or blocked pages. Best-effort generation can still need human correction. Source-local extraction does not solve cross-chunk coreference. Graph search scans the 100 most recent matching runs and at most three quotations/ten neighbouring assertions per hit; it is not yet a persistent vector/graph index. Every candidate still requires scientific review.
