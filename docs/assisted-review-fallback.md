# Assisted review during embedding outages

On 2026-10-19, graph-staging returned HTTP 429 (quota exceeded) from Gemini
embeddings while Ollama text generation remained healthy. The frontend required
both providers, disabling all review actions despite its 75 indexed text segments
and 82 indexed images. Existing vectors cannot embed a new query during an outage.

The review routes now use keyword retrieval from the existing FTS/CJK text index
when embedding readiness fails or a query encounters a transient provider error.
They preserve source ownership, page citations, observation links and citation
validation. English question stopwords are removed for bounded keyword expansion;
up to eight original terms are searched. No new embeddings or indexes are written.
Historical extraction evidence is excluded and material filters remain enforced.

The response is labelled `keyword_evidence_search`; its rank scores are not vector
similarity or relevance probabilities. UI and exports label these as Keyword match.
Image similarity is unavailable in this mode. There is no cross-language semantic
retrieval: use terms present in the original papers, such as 石拍. A no-hit query
does not generate an answer. Generation failures still return an explicit unavailable
response rather than fabricated answers. Normal semantic/image retrieval resumes
when the configured embedding provider recovers.

Endpoints: `/api/v1/repository/ai/semantic-search`, `multimodal-search`, `ask`,
`evidence-report`. `/ai/status` exposes `keyword_review_available` separately from
`embedding_configured`; no billing, credentials or provider configuration changes
are made by this fix.

A separate production incident exhausted its 500 MB volume and prevented durable
checkpoints. The user selected 1 GB in Railway; source files and prior results were
retained. Atomic checkpoint writes now clean up their own failed temporary files
and refuse new writes below an 8 MiB reserve. Existing saved job state remains
untouched on failed writes. This prevents repeated provider work when the initial
checkpoint cannot be written. Free-space monitoring and volume expansion remain
operational responsibilities.

Verification:

```sh
.venv/bin/python -m unittest test.test_processing_storage \
  test.test_repository_indexing_pipeline test.test_processing_jobs \
  test.test_processing_api test.test_document_translations test.test_translation_api -q
```
