# Upload and search review — 2026-10-02

## Cloud state observed

- Vercel production serves frontend commit `bc0d036fc4b8738588f6452bce6b2080553eabcf` on `mvp-ollama-cloud`. The current working branch is `codex/research-progress-portal`; pushing it creates a preview, not a production release.
- The branch preview at commit `2fd8f9df59a7765f883a8158ab107504d93149fb` failed. Its build log reports that the required API URL and Clerk publishable key must be present. The project variable list contains `VITE_API_BASE_URL` for Production and Preview, but lacks `VITE_CLERK_PUBLISHABLE_KEY`.
- Railway's public `/health` returns 200. `/ready` and `/api/v1/progress/status` return 404, indicating that the currently served API does not expose the newer deployment checks and Progress routes. The deployed backend commit, worker, Redis, Postgres, and persistent volume could not be verified because Railway dashboard access was denied.
- The public AI status check reported Ollama Cloud `gpt-oss:120b` generation and Gemini `gemini-embedding-001` embeddings at 768 dimensions as ready. Coverage was 114 of 1,236 text segments and 229 of 682 image records. Blank image artifacts can legitimately remain unindexed; these counts alone do not establish that every image is usable evidence.

Sources: [Vercel production](https://tapa-research-workbench-frontend.vercel.app), [failed preview](https://vercel.com/zacharypangan-s-projects/tapa-research-workbench-frontend/7nDjHwfqQjaSAMyepi39G1WzHsPt), [Railway AI status](https://tapa-research-workbench-backend-production.up.railway.app/api/v1/repository/ai/status).

## Changes implemented

Uploads now save the original file and enqueue a durable extraction job. The worker extracts text, images, and OCR, then prepares both text and image embeddings across successive batches. Queue failures leave the uploaded file intact and expose a retry message. Appending a file preserves existing segment identifiers and citations.

Text and image search use the active embedding provider independently of chat generation. Gemini image vectors are now retrieved with the same model identity used to create them. Exact matching also includes evidence that has not yet been embedded, and remains available when embedding requests fail. The library search includes extracted text, OCR, and image captions.

Image labels can be generated when vision is configured even if embeddings are unavailable. Embeddings can use OCR and nearby source context even if vision is unavailable. Failed descriptions or embeddings are reported as deferred work. Index preparation never deletes blank source evidence. Failed forced caption generation preserves previous captions.

Newly created and rebuilt passage embeddings cover the whole passage through bounded chunks instead of silently discarding its tail. Existing vectors remain compatible with their configured model; reindex with `force: true` to rebuild previously truncated passage vectors.

The workbench displays processing progress, index coverage, supported format guidance, and a Prepare Search action. Related-reference discovery and organized evidence reports remain usable without chat generation. Generated answers still require a reachable chat model.

## Format and offline capabilities

| Source | Prepared evidence |
| --- | --- |
| PDF | Extracted text, embedded images, text-poor scanned page images, OCR |
| PPTX | Slide text and embedded images, OCR |
| DOCX | Paragraph/table text and supported embedded raster images, OCR |
| PNG, JPEG, WebP, TIFF, BMP | Image evidence and OCR; multi-frame raster files are read frame by frame |
| TXT, Markdown, CSV/TSV, JSON/JSONL, VTT/SRT, HTML/XML, RTF | Text passages |
| Other formats, including legacy DOC/PPT and XLSX | Original file and searchable metadata; convert to a supported format for content search |

Extraction, OCR, and exact search run locally on the backend without a cloud AI provider. OCR requires Tesseract; the existing Docker image installs English language data. Other scripts and languages need appropriate OCR language configuration and separate verification. AI image search currently embeds OCR, labels, and source context, rather than raw image pixels.

Local Ollama can supply embeddings, chat, and vision when suitable models are already installed and configured. Offline operation means a reachable local backend and local services; it does not add a browser offline cache or make a Railway-hosted API reachable without Internet access. Cloud providers require network access. Actual local model inference and cloud image captioning were not verified in this review.

Files retain the 100 MB upload limit and extraction retains a configurable passage limit, up to 5,000 per run. Limit warnings identify remaining files or truncated source contents. Larger sources may require splitting. Long indexing jobs retain the existing worker timeout; indexed evidence persists and Prepare Search can resume missing work after a failure.

## Verification and release requirements

Regression tests cover automatic upload jobs, queue outages, append behavior with citations, indexing across batches and forced rebuilds, provider contracts, unavailable providers, Gemini image retrieval, vision without embeddings, and supported formats. Real Tesseract OCR was checked with synthetic PNG and scanned PDF sources. The local browser flow was checked from upload through worker completion to retrieved evidence, with deterministic embeddings and chat disabled. Frontend lint and a production build with synthetic configuration pass.

Before a production release, configure the real Clerk publishable key in the appropriate Vercel environments and verify backend authentication configuration. Confirm Railway's deployed commit and shared persistent storage, database, Redis, and worker. Release the intended commits through the configured production branches, then test an authenticated synthetic upload and indexing job. Existing production research files were not uploaded, modified, or bulk reindexed during this review.

The `/repository/ai/index` endpoint now returns HTTP 202 with a job ID, like the extraction and image-index endpoints. API consumers must poll `/progress/jobs/{job_id}` rather than expect synchronous indexing counts.
