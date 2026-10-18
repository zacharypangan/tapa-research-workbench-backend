# English reading copies

The English translation action creates a machine-generated reading copy of the
current attached PDF, DOCX or text documents. Original file bytes, extraction
units, source quotations, search indexes and graph evidence remain unchanged.
Translations require human review, especially OCR text, names, chronology and
technical archaeological terms. A successful request proves coverage and
provider completion, not linguistic or scientific accuracy.

## API

All routes use the existing `/api/v1/repository` prefix and existing application
access policy. No new authentication configuration is introduced.

1. `POST /materials/{material_id}/translations` with `{}` translates all supported
   attachments (maximum 20). Pass `{"file_id":"..."}` to select one attachment.
   Returns HTTP 202 and an idempotent job descriptor.
2. `GET /materials/{material_id}/translations` lists jobs, newest first, with
   `stale` marking changed source bytes, removed files or changed translation
   implementation. `GET /translation-jobs/{job_id}` returns saved progress.
3. `GET /translation-jobs/{job_id}/artifact` returns all original units and
   completed English reading copies, including partial results while running.
   `?format=text` downloads a clearly labelled English reading copy. Failed and
   pending units appear as explicit placeholders rather than omitted pages.
4. `POST /translation-jobs/{job_id}/retry` retries unfinished units. Successful
   checkpoints are reused unchanged. If source bytes or implementation changed,
   retry creates the current job and links the historical job to its replacement.

Statuses: `queued`, `running`, `done`, `partial`, `error`, `superseded`. Coverage
contains `total`, `done`, `failed`, `pending`. `done` requires every readable unit
translated and the source representation to have complete readable coverage.
Unreadable pages, omitted extraction units or partial OCR keep the job `partial`.

## Reproducible method

- Hash source bytes and record file identity, language, current OCR/runtime and
  representation signature. Use the same content-addressed source transcription
  used by original-document search and graph extraction.
- Divide each complete page/paragraph into contiguous chunks of at most 2,000
  Unicode characters. Offsets cover every character exactly once. Record the
  original locator, original text, offset range, original SHA-256, file SHA-256,
  representation ID and OCR confidence/method.
- Translate each chunk with the existing configured Ollama text model. Source
  content is delimited as untrusted material. The system prompt requires full
  translation without added facts, summary or inferred missing OCR content.
- Requests permit 4,096 output tokens, a 120-second timeout and at most 128 KiB
  response bytes. GPT-OSS uses its supported `low` thinking setting. Provider
  reasoning traces are never stored. A response must indicate completion and
  contain English text; length-terminated output is rejected.
- Persist each successful unit immediately on the existing Workbench volume in
  `translations/{job_id}/{unit_id}.json`. Record model, prompt version,
  implementation hash, time, provider completion counters and response hash.
  Missing or mismatched checkpoint hashes cannot count as completed.
- Two consumers bound translation concurrency. Shared-storage locks prevent two
  API processes from executing the same job. Up to two requests are attempted
  for transient failures. Provider outages and configuration failures stop the
  remaining job after that bound; retry resumes saved work. Interrupted jobs
  return to the queue, and queued/running disk jobs resume on backend startup.
- Check source freshness before work, before each unit, and before completing the
  job. Changed-source results remain historical and cannot become a current
  complete translation. Artifacts/listing independently mark stale results.

The translation is not automatically an accepted knowledge graph assertion or
original quotation. A separate derived-document workflow would need explicit
machine-translation provenance and links back to these source locators.

## Verification

Run focused deterministic tests without provider access:

```sh
.venv/bin/python -m unittest test.test_document_translations test.test_translation_api -q
```

Tests cover continuous source coverage, provenance, idempotency, source/model
changes, stale versions, partial OCR, checkpoint mismatch, bounded failures,
missing provider, truncated/oversized output, cancellation, retry reuse,
partial artifact access and English download labels. Live language quality
still requires comparing the resulting English copies with the original papers.
