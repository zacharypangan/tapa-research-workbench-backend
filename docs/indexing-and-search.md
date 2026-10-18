# Full-coverage indexing and search

The automatic processing worker calls the existing `index_missing_segment_embeddings(con, material_id=None, limit=200, force=False)` and `index_missing_image_embeddings(...)` helpers in bounded batches. It must continue while `pending_count` decreases, and retain `outcomes`/`failed_ids` when an item needs retry. Both helpers accept optional `skip_ids` to finish all other healthy rows before reporting a failed-item union; the filter applies before the batch limit. `pending_count` intentionally still includes excluded failed rows so failures cannot masquerade as complete coverage. Search does not prepare missing embeddings unless its explicit `auto_index` flag is enabled.

## Storage and deterministic replay

No database schema or provider setting changes are needed. Existing embedding JSON columns store `repository-span-embeddings-v2` objects rather than one unversioned vector list. The stored `model` key is `active_embedding_index_key()`:

```
<provider/model/dimensions identity>|spans-v2:2000:160:RETRIEVAL_DOCUMENT
```

Every parent string is divided into spans of at most 2,000 characters with 160-character overlap. Whitespace boundaries are preferred; contiguous Chinese/Japanese text is fully covered without requiring whitespace. These are conservative character bounds, not a claim of exact tokenization. Offsets are zero-based, end-exclusive Python string offsets. SHA-256 hashes cover each exact span and the complete parent text. Every span must have a finite, nonzero, dimension-consistent vector before a parent is marked complete. Provider adapters no longer silently cut input at 2,000 characters; the dispatch rejects oversized raw requests so callers must split explicitly.

Text parent offsets refer to `extracted_segments.content_text`. Image parent offsets refer to the normalized combination of separately retained OCR, generated visual caption and human observation labels. Image retrieval remains text-based evidence retrieval, not image-pixel similarity. A machine description retains its own evidence status; indexing it does not turn it into an original quotation.

Replay skips only a complete current-format/current-model record with matching content hash and offsets. Changed content and old-method vectors are reindexed incrementally. Legacy vector lists remain searchable while migration runs, explicitly marked `index_legacy=true`; they do not count toward full-coverage readiness. Current semantic ranking takes the maximum cosine similarity across a parent's spans and returns `matched_span`/`matched_text` so a tail match is not lost when creating an answer. Original parent text and citation locators remain available.

Provider calls run with concurrency three per indexing operation, transient failures retry at most three times with short exponential delays, and identical span hashes share requests within a batch. Network operations do not hold write transactions. Each successful item commits independently; an item failure does not discard successful neighbours. Before writing, the service checks that the source/annotations have not changed. Source images are retained; decorative/blank images become explicit skip records and no longer block the queue. Missing image files and unusable OCR/description are visible failures rather than permanent successful skips. Superseded rows retained for human observations use `historical_` source kinds and are excluded from current indexing/retrieval.

`embedding_coverage(con, material_id=None)` reports eligible, current, missing, stale and skipped rows separately for text/images. It counts complete spans, not just a matching model string. A successful provider health probe does not imply corpus completeness.

## Description provenance and freshness

Machine descriptions are cached on the existing volume at `derived/image_descriptions/<prefix>/<key>.json`. The key hashes image-byte SHA-256, exact supplied context SHA-256, the vision model and `observable-image-labels-v3` prompt version. Raw model text, normalized labels, status, timestamps and bounded attempt history are retained separately from human observations. Failure records store safe categories (authentication, rate limit, network, malformed response or no usable observable labels); provider error bodies and credentials are not saved. Successful caches and in-flight deduplication prevent duplicate image calls. Failed descriptions remain retryable.

New image indexes pin their description recipe and caption hash. Legacy captions refresh once; changed image/context/model/prompt invalidates current description readiness. Description failure preserves the previous caption for inspection and leaves processing visibly incomplete. If fresh labels are identical, valid existing text-span vectors are reused rather than paying to embed unchanged text again. File-stat-guarded image hash caching avoids re-reading unchanged image bytes on every status/search request. The preparation job fingerprint includes the vision model/prompt version so service upgrades schedule new work.

## Query processing

`GET /repository/search?q=...` uses literal mode by default: each term is FTS-quoted, terms use AND semantics, and punctuation/FTS operators cannot execute as query syntax. Contiguous CJK strings use an additional literal substring fallback, including two-character queries, under the same material and current-source filters. `advanced=true` explicitly enables FTS5 expressions; malformed expressions return HTTP 400. No tokenizer migration is required. Results retain BM25 score/basis where available; CJK substring hits are labelled separately. Graph entity results are a separate `graph_results` channel with graph/evidence/review provenance.

Semantic requests default to hybrid retrieval. Lexical candidates and semantic parent-span candidates are fused using reciprocal rank fusion with k=60. `hybrid_score`, `lexical_rank`, `lexical_score`, `semantic_rank` and cosine `semantic_score` remain distinct. They are ranking measures, not calibrated scientific relevance probabilities. Lexically matched passages remain discoverable before their vectors are ready. Set `hybrid=false` to compare against the maximum-span-cosine baseline. Full vector scans remain appropriate for the current small corpus; add an approximate vector service only after measured latency/recall warrants it.

Text/image branches reuse one query vector, with a 128-entry five-minute in-memory cache and in-flight request deduplication. Provider/model identity and query content key the cache. Readiness probes are cached for 60 seconds after success and 10 seconds after failure; generation and embedding readiness are independent. Corpus answers refuse unavailable generation with HTTP 503 before attempting an answer. Numbered citations outside the supplied passage range cause the generated answer to be withheld. Responses explicitly label model-generated draft text and return `answer_validation` flags; citation bounds are a structural check, not factual entailment validation. Missing numbered citations are visible in those flags and all generated claims remain subject to review. Caches are performance aids and can be discarded on restart without affecting stored evidence.

## Reproduction

```sh
.venv/bin/python -m unittest test.test_repository_indexing_pipeline test.test_repository_image_search test.test_image_descriptions -v
```

Fixtures cover complete beginning/middle/end spans, tail retrieval, content invalidation, legacy migration, retry/partial success, duplicate-call reuse, provider concurrency, long image representations, explicit skips, missing image failures, annotation edit races, source-history filtering, Gemini/Ollama image round trips, literal malformed expressions, two-character CJK AND/filter behavior, unembedded hybrid evidence and independent readiness/query caching. Providers are mocked; this verifies data flow and integrity, not live model retrieval quality.

Before changing retrieval thresholds, evaluate a frozen multilingual query/source fixture with human relevance labels (Recall@10/nDCG@10 and source-quote accuracy), latency and provider call counts. Preserve separate raw cosine and fusion measures in reports.
