# Research knowledge graph service

Runnable document ingestion, grounded concept and relationship extraction, document-seeded web collection, review and immutable graph export. The API is mounted at `/api/v1/research-runs` and uses the existing hosted Workbench's shared repository access. Graph jobs run in the API process and persist their state on the repository volume; they do not require Progress, ARQ, an additional database, or a separate worker.

See [the hosted integration and reproducible methodology](../../../docs/knowledge-graphs.md).

## Operator quick start

Run from `stling_backend`. No installation is performed by these commands.

For a complete demonstration without network or model calls:

```sh
.venv/bin/python -m scripts.knowledge_pipeline_example --output /tmp/research-kg-offline-example
```

Choose a fresh directory. This produces a clearly labelled synthetic pending preview with two sources, six entities, two reported relationships and two seed-bound alignments. Its Python annotations are authored fixtures, not a semantic extractor. Use the following CLI/API workflow for real materials.

```sh
.venv/bin/python -m app.repository.knowledge_pipeline \
  --root /tmp/research-kg-demo --organization local-demo \
  create /path/to/new-paper.pdf --config /path/to/config.json \
  --idempotency-key first-paper-version
```

The optional configuration merges with the bundled tapa ontology 0.2.0:

```json
{
  "research_topic": "tapa barkcloth manufacture archaeology",
  "processing": {"mode": "ollama", "batch_chars": 12000},
  "augmentation": {"enabled": true, "max_pages": 5, "delay": 1, "timeout": 20}
}
```

Use the printed UUID as `RUN_ID`. `advance` ingests and processes until an explicit input/review gate:

```sh
.venv/bin/python -m app.repository.knowledge_pipeline \
  --root /tmp/research-kg-demo --organization local-demo advance RUN_ID
```

Ollama mode reuses the Workbench's existing provider configuration and authorization helper. It sends structured extraction requests; it does not derive claims from headings or filenames. An unavailable configured provider preserves checkpoints and fails the current stage; unprocessed failed extraction batches can also receive submitted annotations before resume. Manual mode (the default) creates `primary_batches.json` and pauses at `awaiting_annotations`. Each batch includes exact unit IDs/text; `schema RUN_ID` prints the extraction response schema. Submit a JSON object keyed by batch ID using `annotate RUN_ID responses.json`, then `advance` again. Processed annotations are immutable; corrections require a new run.

Rejected model output pauses the affected batch at `awaiting_annotations`, with its raw response and validation reasons preserved under `rejected_batches/`. It is never coerced into claims or counted as a successful empty extraction. Submit a validated annotation for that unprocessed batch and resume; rejected responses remain available for audit, and already valid batches retain their checkpoints.

At `awaiting_frontier`, inspect `seeds.json`. Register URLs against those IDs:

```json
[
  {
    "url": "https://public-source.example/page",
    "seed_ids": ["DOC_001:entity:ToolForm:ACTUAL_ID_FROM_SEEDS"],
    "question": "Which reported tool forms and production contexts extend this paper's account?",
    "query": "the executed discovery query",
    "discovery_method": "researcher_search",
    "discovery_locator": "the discovery page or search record",
    "selectors": "main article"
  }
]
```

Use `frontier RUN_ID frontier.json`, then `advance RUN_ID`. The service generates query plans; a researcher or search connector supplies discovery results. It does not execute search-engine queries automatically. URLs cannot be added independently of primary seeds. Collection is depth zero, public hosts only, robots-aware, page-bounded and DOM-audited. Selectors are optional; narrow them to the relevant page content where possible. If no selector is given, the collector's extraction rules apply and quotations still require relevance review.

The worker lazily reuses an installed scraping interpreter. Locally it can find the previously approved `tmp/tapa-web-pilot-runtime-20261009/bin/python`; it never installs dependencies. On a host, provision the existing pilot requirements in the worker's interpreter, or select an existing interpreter for CLI runs with the top-level `--collector-python` option. API callers cannot select executables. The service uses static HTML/BeautifulSoup by default; the underlying adapter also supports the existing Selenium/Playwright/Trafilatura pilot for operator-driven collection. Those browser options are not exposed by the current HTTP configuration.

At `awaiting_review`, inspect `graph_input.json`, source snapshots, quotations, `primary.json`/`augmented.json` rejection lists and `augmentation_metrics.json`. Decisions are a list:

```json
[
  {"candidate_id": "actual entity/assertion/alignment ID", "decision": "accept", "rationale": "Reason for inclusion"}
]
```

Use `review RUN_ID decisions.json` and `release RUN_ID`. A release contains accepted candidates and excludes pending/rejected candidates. An accepted relationship requires accepted endpoints and scope entities. `release RUN_ID --preview` includes pending candidates with their status and leaves the run open for review. Inclusion approval never asserts scientific truth; source claims and analytical correspondences retain their evidence, polarity, context and pending specialist status.

## HTTP and hosting

The API accepts registered `material_id`/`file_id` pairs, not arbitrary paths. Continue using the existing upload route first, without calling the separate repository extraction route when this pipeline will do extraction. Then:

```json
POST /api/v1/research-runs
{
  "documents": [{"material_id": "registered-material", "file_id": "registered-file"}],
  "config": {"processing": {"mode": "ollama"}, "augmentation": {"enabled": true}}
}
```

Send an `Idempotency-Key` header to reuse identical input snapshots. The response supplies the run ID and durable job ID. Poll `GET /api/v1/research-runs/{run_id}`; the graph status is the authoritative execution and input-gate status. A completed queue job can mean `awaiting_annotations`, `awaiting_frontier` or `awaiting_review`; inspect the run's status before treating the research work as complete.

| Operation | Route suffix after `/research-runs/{run_id}` |
|---|---|
| Inspect state and artifact inventory | empty |
| Download exact artifact | `/artifacts/{relative_name}` |
| Submit source-grounded annotations | `/annotations`, body `{"responses": {...}}` |
| Register seed-bound web pages | `/frontier`, body `{"frontier": [...]}` |
| Resume processing | `/resume` |
| Record inclusion decisions | `/review`, body `{"decisions": [...]}` |
| Compile release or preview | `/release`, body `{"allow_pending": false}` |
| Continue with documents only | `/skip-enrichment`, body `{"reason": "..."}` |

`GET /research-runs` lists saved projects with bounded pagination. `GET /research-runs/capabilities` reports configured model/collection support without exposing model credentials. The browser connects these routes through a single Knowledge graph workspace.

Start the existing API normally. Its startup hook starts the graph scheduler; queued jobs and jobs interrupted by a process restart resume from checkpoints. Every scheduler sharing the repository volume uses advisory execution locks so the same run cannot execute twice at once. A failed job remains failed until an explicit resume. Persist `REPOSITORY_STORAGE_ROOT` on the existing Railway volume; an ephemeral filesystem cannot preserve source snapshots or releases.

The HTTP integration preserves the hosted MVP's shared access model. It does not add per-user tenancy or roles. Recorded actor names are shared-workbench attribution, not authenticated personal identities. Existing platform access protection continues to govern this staging deployment.

The graph remains separate from the legacy generated graph tables and search index. This branch adds no Progress, Atlas or Governance routes and requires no database migrations.

## Verification

```sh
.venv/bin/python -m unittest \
  test.test_knowledge_pipeline_adapters \
  test.test_knowledge_pipeline_processing \
  test.test_knowledge_pipeline_contracts \
  test.test_knowledge_pipeline_graph \
  test.test_knowledge_pipeline_runtime \
  test.test_knowledge_pipeline_service \
  test.test_research_runs -v
```

Run adapter tests in the approved collector interpreter to exercise its optional real-collector fixture. Tests use controlled sources and mocked provider/network responses; they establish software behavior, not archaeological extraction precision. These tests do not contact a live model or public website.
