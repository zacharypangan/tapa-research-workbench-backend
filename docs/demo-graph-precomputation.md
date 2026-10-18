# Saved demonstration graphs

Graph creation can save a useful draft after a bounded pass while retaining every source passage. Set `processing.max_auto_batches` to an integer from 1 through 100. Omit it or use `null` for the existing unlimited behavior. The limit counts fresh provider **batches**, including their bounded repair attempts, across primary and web extraction within one invocation. Reused, validated checkpoints and submitted annotations do not consume this budget.

When the limit defers a batch, the run remains `awaiting_annotations`, with `automatic_batch_budget_reached: true`, `deferred_batch_ids`, all `missing_batch_ids`, and any `rejected_batch_ids`. `validated_preview.json` is explicitly partial. No page or batch is removed, and no release or scientific acceptance is implied. `POST /research-runs/{run_id}/resume` advances the next budgeted pass. Rejected batches retain their audit and require corrected annotations or a derived retry; resume does not repeatedly call the provider for rejected evidence.

Automatic preparation creates drafts with a two-batch pass. Explicit graph requests retain their requested/default limits, and existing runs are unchanged. The graph list exposes `has_saved_candidates` so navigation can open a saved draft while newer work continues.

## Reuse registered extraction for any original format

`POST /research-runs/from-extraction` accepts:

```json
{
  "material_id": "registered-material-id",
  "file_ids": ["registered-file-id"],
  "title": "Demo saved draft — document title",
  "config": {
    "processing": {"mode": "ollama", "batch_chars": 4000, "max_auto_batches": 2},
    "augmentation": {"enabled": false}
  }
}
```

Omitting `file_ids` selects all files belonging to the material, up to 20. Every selected file must have readable current registered extraction segments. The route snapshots **all** those segments into a content-addressed JSON representation in managed storage, then registers that immutable representation in the run's inputs. CSV, PPTX and RTF originals keep their original bytes and database records. They are not mislabeled as PDFs or plain-text uploads.

The graph source is `source_kind: registered_extraction` with `derived_from` containing the original file locator, filename, SHA-256 and byte count. Its snapshot hash identifies the derived JSON bytes, not the original file. Each unit retains the original segment ID, source kind, page/row/slide reference, page index, extraction timestamp and original file hash. Only exact `content_text` enters concept extraction; filenames and provenance metadata do not become passages. The latest material extraction record is attributed explicitly as material-wide metadata, rather than asserting a per-segment extraction run that the current schema does not record. Transcription correctness still requires source review.

Original PDFs may also use this action to reuse completed OCR/extraction without parsing the same bytes again. The standard document-ingestion action remains available.

## Reproducible warmup

The CLI defaults to a read-only plan; add `--execute` to queue drafts. It reuses nonempty saved graphs and current bounded automatic drafts before creating another run. It saves a report after each request, uses idempotency keys, prioritizes short existing research PDFs, and reports unextracted documents instead of silently skipping their content.

```sh
.venv/bin/python scripts/precompute_document_graphs.py \
  --api https://YOUR-BACKEND/api/v1 \
  --output ../SampleReports/demo_graphs_2026-10-19/live_drafts.json \
  --execute --max-auto-batches 2 --batch-chars 4000 --timeout-seconds 600
```

Repeat `--material-id ID` to target particular sources. Use a new `--key-prefix` after intentionally changing the pinned implementation or request. The report contains IDs and counts, without passages or credentials. A zero-entity draft is reported honestly and may need another explicit pass; a nonempty graph is still unreviewed.

Check a selected graph's exact quotations and ownership with:

```sh
.venv/bin/python scripts/verify_document_graph_run.py \
  --api https://YOUR-BACKEND/api/v1 --run-id RUN-ID --output verification.json
```

Focused verification:

```sh
.venv/bin/python -m unittest test.test_knowledge_pipeline_service test.test_research_runs \
  test.test_processing_jobs test.test_knowledge_pipeline_graph test.test_knowledge_pipeline_contracts -q
```
