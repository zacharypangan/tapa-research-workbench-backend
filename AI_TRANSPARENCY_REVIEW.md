# AI transparency release — 2026-10-05

## Delivered behavior

- Exact-search fallback results carry `retrieval_method=exact_text` and a null
  semantic score. Cards and PDF/CSV exports show retrieval method or raw embedding
  similarity instead of a percentage that could be mistaken for accuracy.
- Image results distinguish literal matches in OCR, AI descriptions, researcher
  observation labels, and metadata. An AI-description match is not labeled direct
  image text evidence. OCR and AI descriptions retain separate labels when copied
  into observations and exports.
- `GET /api/v1/repository/ai/coverage` reports extraction and current-model index
  coverage for the repository, optionally scoped with `material_id`. It uses stored
  data without contacting a provider. Counts include files without extracted
  evidence, failed/unsupported files, and latest material extraction failures.
- Repository answers return the exact input excerpts, citation numbering,
  truncation, provider/model, and whether images were supplied. The UI exposes these
  details next to the generated draft. Current answer generation uses text passages.
- Progress evidence inspection shows quotations, translations, locators, evidence
  relations, and navigation to repository sources. Existing AI run metadata loads
  on demand and is returned only when the run actually lists the candidate record.
- Review notes, reviewer identity, timestamp, and recent decision/note history are
  visible. Existing audit storage records note changes and explicit clearing.
- Atlas normalization preserves Unicode scripts and accents using NFC. Ordinary
  ASCII identity hashes stay compatible. Identifiable legacy lossy records remain
  available with an identity review notice; corrected identities do not inherit
  possibly conflated decisions. Existing rebuilds regenerate graph relationships.

## Verification

Backend regression command:

```sh
.venv/bin/python -m unittest test.test_repository_transparency test.test_upload_indexing test.test_semantic_graph test.test_progress test.test_production_fixes
```

Frontend checks:

```sh
node --test test/retrieval-transparency.test.mjs
npm run lint
npm run build
```

The production build requires API and Clerk public configuration. Compilation was
verified with synthetic public values; no environment files or deployment settings
were changed. Generated build output is excluded from the commit.

Browser verification used isolated temporary databases, synthetic source data, and
mocked generation. Checked offline exact-match labels, whole-repository and selected
material coverage, truncated generation excerpts, bilingual quotations, saved review
notes/history, and navigation back to the cited repository source. This does not
evaluate a live model's scholarly accuracy or verify cloud deployment.

## Limits and rollout

Deploy the backend before the frontend to enable the additive coverage/provenance
endpoints. Older backends remain usable and show unavailable-detail notices.
No dependency, database schema, authentication, or deployment configuration changes
are required. Source data and saved review decisions are retained.

Coverage is a current aggregate, not a frozen corpus manifest or an OCR/language
quality assessment. Repository answer input metadata is returned per response; it
is not a persisted generation ledger. Progress uses its existing persisted AI runs.
Citation entailment and image descriptions still require researcher verification.

Previously conflated entity forms cannot be reconstructed automatically. Retained
legacy entities preserve references, while generated relations/resolutions follow
the existing rebuild behavior; those identifiers are not migrated. A reviewed
reconciliation remains necessary for historical collisions that cannot be identified
from the stored label. This release adds no transliteration or alias workflow.
