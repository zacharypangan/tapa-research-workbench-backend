# Document graphs in the hosted Workbench

This branch starts from `mvp-ollama-cloud` and adds only document graph creation. Existing repository, search, extraction and reporting flows keep their existing routes. Graphs use one additional workspace and `/api/v1/research-runs` API. Progress, Atlas, Governance, organization databases and new authentication flows are not included.

## Replicable method

1. **Register documents.** Create a material and upload PDF, TXT, Markdown, JSON or DOCX through the existing repository routes. Create a graph with registered material/file pairs; filesystem paths cannot be supplied through the API.
2. **Freeze inputs.** The service copies each file into the graph project, records its SHA-256, original repository locator, ontology version/configuration and pipeline implementation hashes. Input limits are 20 documents, 50 MiB per document and 100 MiB per graph.
3. **Extract readable units.** PDF pages, document paragraphs, text spans or JSON leaves become units with explicit locators and text hashes. Empty/scanned documents pause for readable input rather than inventing evidence. Original-language text is preserved.
4. **Process concepts and relationships.** Manual annotations or the existing configured Ollama provider produce structured responses in bounded source batches. Validation requires exact quotations from registered units, permitted ontology types and predicates, valid endpoints, polarity and contextual scope. Invalid claims are recorded as rejections. Saved provider responses/checkpoints support audit and resume.
5. **Optionally augment from document concepts.** The primary concepts generate research seeds and suggested queries. A researcher registers public URLs against those seed IDs. Static collection records bounded, robots-aware snapshots, hashes and observed-page locators. Collection does not search the internet automatically. Web concepts retain separate identities; evidence-backed correspondence proposals connect them only to the registered document seeds. A recorded skip decision allows a document-only graph.
6. **Review evidence.** The preview exposes source quotations, original locators, claims, contextual qualifiers and correspondence proposals. Accepting inclusion is separate from archaeological or specialist validation. Pending specialist status remains explicit.
7. **Publish and export.** A final release contains accepted candidates only. Accepted relationships require accepted endpoints and relevant context. JSON, JSONL, qualified RDF/Turtle, provenance and a hash manifest are written atomically. A pending preview is separately identified and does not close review. Published releases are immutable; changed materials require a new graph.

## Storage and execution

Graph files live beneath the existing `REPOSITORY_STORAGE_ROOT/research_runs/<workspace-hash>/<run-uuid>/`. This hosted MVP uses a fixed shared-workbench namespace, matching the existing shared repository. It does not claim per-user isolation or personal audit identities.

The API starts one graph scheduler per API process. Enqueue writes hidden job metadata and the public run state to the persistent volume. A separate execution lock permits only one scheduler to process a run. The pipeline also locks run mutations. Startup rediscovers queued/running jobs and resumes immutable stage checkpoints. Transient lock contention remains queued; extraction failures preserve checkpoints and require explicit resume. Shutdown cancellation records the job for recovery. A queue job can finish at an input gate while the research graph remains incomplete.

Use the backend's existing persistent Railway volume. No Progress database, schema migration, Redis job queue, ARQ worker or additional service is required for graph processing. The frontend preview must target a backend containing these graph routes; deploying only the frontend does not install an API.

The model uses the existing repository generation configuration and authorization helper. No provider credential is written into graph artifacts. Manual processing works without model credentials. Static web augmentation requires BeautifulSoup and httpx in the backend interpreter; browser automation packages are optional operator tools and are not required for document graphs.

## HTTP flow

```text
POST /api/v1/repository/materials
POST /api/v1/repository/materials/{material_id}/files?filename=paper.pdf
POST /api/v1/research-runs
GET  /api/v1/research-runs/{run_id}
POST /api/v1/research-runs/{run_id}/annotations  # when manual input is needed
POST /api/v1/research-runs/{run_id}/frontier     # optional, seed-bound URLs
POST /api/v1/research-runs/{run_id}/resume
POST /api/v1/research-runs/{run_id}/review
POST /api/v1/research-runs/{run_id}/release
GET  /api/v1/research-runs/{run_id}/artifacts/{relative_name}
```

Create body for an initial document-only graph:

```json
{
  "title": "Barkcloth tool evidence",
  "documents": [{"material_id": "registered-material", "file_id": "registered-file"}],
  "config": {
    "research_topic": "tapa barkcloth manufacture archaeology",
    "processing": {"mode": "manual"},
    "augmentation": {"enabled": false}
  }
}
```

An `Idempotency-Key` header reuses the same input snapshot only when the request fingerprint matches. Poll run status until an input gate, failure or release; HTTP request completion is not proof of completed extraction. The browser supplies the annotation/review forms, so researchers do not need to write JSON.

The bundled tapa ontology is the default. Programmatic callers can supply another explicitly versioned ontology with entity types and predicate constraints. Use a new run for ontology revisions; unfinished runs require their pinned implementation version.

## Verification

```sh
.venv/bin/python -m unittest \
  test.test_knowledge_pipeline_adapters \
  test.test_knowledge_pipeline_contracts \
  test.test_knowledge_pipeline_graph \
  test.test_knowledge_pipeline_processing \
  test.test_knowledge_pipeline_runtime \
  test.test_knowledge_pipeline_service \
  test.test_research_runs -v
```

The API tests upload a real temporary text file through the existing repository, create a graph, submit grounded annotations, inspect evidence, review, publish and download exports. They also verify idempotency, input/artifact validation, saved project listing, restart recovery and duplicate-execution locks. Controlled provider/web fixtures verify the document-seeded augmentation connection separately. This establishes software behavior; scholarly extraction accuracy still requires material-specific review.
