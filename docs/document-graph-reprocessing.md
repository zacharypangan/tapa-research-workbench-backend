# Replicable hosted document graph backfill

## Automatic document processing

The existing extraction/preparation hook queues graph creation after readable
text has been saved. New assisted runs bind the versioned provider in
`app/repository/document_graph_candidates.py` through
`app/repository/document_graph_runtime.py`. Preparation uses registered text,
1,500-character batches and no automatic batch-count cutoff. It checkpoints
each batch; manual configurations and existing immutable runs keep their prior
protocol. The command below invokes exactly this hosted provider.

Each bounded passage goes through exact mention selection, ontology typing,
relationship extraction, one missing-relationship recall pass and semantic
diagnostics. Server-issued IDs and original source offsets anchor the output;
the unchanged full evidence/ontology validator checks candidates. Missing
referents, saturated allowances and failures remain explicit. Recall does not
invent new entities; unresolved entity questions require another extraction or
researcher input. A model's verification is a diagnostic, not acceptance.

The request policy includes 16,384 context tokens, 4,096 output tokens and
bounded calls/windows. GPT-OSS requests low reasoning effort; other supported
models request thinking disabled. See the provider's versioned `LIMITS` for
exact budgets. Model digests are recorded when supplied; cloud tag-only results
explicitly report that a digest was unavailable. A quota/availability failure
is checkpointed before further batch calls pause, and low checkpoint storage
stops inference. Source blocks retain 64 characters of exact overlap to make
boundary referents selectable. Context is bounded; cross-batch coreference and
semantic completeness still need evaluation.
The backend code must be deployed before the CLI is used. Existing service
settings and database schema remain sufficient.

## Where execution lives

Railway hosts the API and durable graph scheduler on its existing persistent
volume. API startup discovers queued or interrupted `.job.json` records; saved
batch checkpoints prevent completed work from being repeated. A per-run
execution lock prevents two consumers from processing the same run on shared
storage. The CLI may run on a researcher's computer or another administration
host: it submits and monitors server jobs rather than performing extraction.
Closing it does not cancel an accepted hosted job. Availability failures and
validation stops still require explicit attention or resumption.

Inference goes to the backend's configured provider. The current production
rollout uses Ollama Cloud; installing this script does not redirect production
to Minpaku. A configured, available Minpaku worker can handle inference through
the existing worker bridge. Model stages run sequentially within a passage,
while the graph scheduler has two consumers **per API process**. Translation,
OCR and other processing schedulers have their own work, so two graph consumers
are not a global server or GPU concurrency cap. Additional API replicas would
also add consumers. The Minpaku pilot currently enforces unloading after each
request (`keep_alive=0`); keeping a model warm across stages needs a separately
reviewed resource policy before high-throughput self-hosting.

Before each staged passage, the runtime checks free space on the checkpoint
volume against a conservative allowance:

`max(128 MiB, 2 jobs × max_calls × (4 × response_bytes + 2 × prompt_bytes) + 16 MiB)`

The fourfold response allowance covers the persisted provider envelope and
parsed stage response, with a twofold allowance for pretty-printed JSON. Request
audits receive their own allowance. Stage responses are capped at 128 KiB;
larger responses are explicitly rejected before their bodies enter saved audits.
With current limits, the 128 MiB minimum applies before starting another
passage. The check adapts to the provider's
versioned call/response limits and leaves existing checkpoints available when it
stops. It is an admission check, **not a disk reservation or safety guarantee**:
other services, replicas, atomic rewrites and unusually expansive JSON can still
fill a shared volume. Full-library processing has no aggregate storage or model
call budget; monitor actual volume use and reported failures throughout a
backfill. This check does not expand the user's storage allocation or remove
audit artifacts.

## Backfill command

`scripts/reprocess_document_graphs.py` applies the deployed staged document-claim
provider to the full current registered extraction of every eligible uploaded
file. It uses the existing Workbench APIs and needs only Python's standard
library. It does not upload originals again or invoke a model directly.

The result is **graph candidates awaiting researcher review**. Processing every
registered passage does not establish complete information extraction, correct
interpretation, scientific acceptance or improved accuracy. This command never
reviews, publishes, merges networks, deletes old graphs, scrapes new pages or
renews an external source's eligibility or retention.

## Run a plan, a canary, then the library

Replace `https://HOST/api/v1` with the deployed API base, including `/api/v1`.
These examples contain no credentials. The API's ordinary document/graph access
policy applies; the CLI introduces no authentication changes or private model
gateway access.

```sh
# Read APIs only; save a full inventory and plan locally.
python3 scripts/reprocess_document_graphs.py \
  --api-base https://HOST/api/v1 \
  --output-dir /absolute/path/backfill-plan --dry-run --all-batches

# Separate small canary: wait for its terminal status and inspect its candidates.
python3 scripts/reprocess_document_graphs.py \
  --api-base https://HOST/api/v1 \
  --output-dir /absolute/path/backfill-canary \
  --material-id MATERIAL_ID --execute --all-batches

# Full library: durably queue all eligible files; return after submission.
python3 scripts/reprocess_document_graphs.py \
  --api-base https://HOST/api/v1 \
  --output-dir /absolute/path/backfill-plan \
  --execute --resume --all-batches --queue-only

# Monitor those same runs; omit --queue-only and preserve the method flags.
python3 scripts/reprocess_document_graphs.py \
  --api-base https://HOST/api/v1 \
  --output-dir /absolute/path/backfill-plan \
  --execute --resume --all-batches
```

`--queue-only` requires `--execute --all-batches`. Its `queued_for_processing`
result explicitly means unfinished work. The hosted durable scheduler owns these
jobs after the CLI exits. Backend availability, worker availability and the
pipeline's validation/failure stops still determine whether each job finishes.
Restarting this CLI is not a substitute for checking failed or rejected batches.

Without `--all-batches`, the default is four automatic batches per server pass.
The CLI polls, saves progress and resumes deferred passages until completion or
a documented stop. That mode needs the CLI to remain running for additional
passes. It is useful for a bounded canary but is less suitable for queuing large
books. Both modes default to 1,500 characters per passage batch. The deployed
provider may impose its own stricter constraints.

`--dry-run` is the default. `--resume` explicitly reopens an existing local
manifest; existing output is otherwise protected from accidental replacement.
Use the same `--all-batches`, `--batch-chars` and `--max-auto-batches` settings
when continuing a run. Changing method settings or a source creates a new
identity and preserves earlier entries in the manifest's history.

## Identity, full coverage and reuse

Before submitting work, the CLI:

1. Requires configured automatic extraction, registered-extraction support and
   the `staged_document_claims` candidate method. It checks the provider version,
   code hash, explicit researcher-review requirement and pinned pipeline digest.
   The ontology must bind the same provider code/version and execution identity
   (model, provider hash and runtime code hash). An older deployment fails closed.
2. Enumerates **all** materials using `limit=500` and `offset`, then all existing
   material-associated runs using `limit=100` and `offset`. It never interprets
   the first page as the whole database. A changing total, duplicate material
   list or truncated listing stops planning rather than silently losing files.
3. Gets every material's read-only extraction inventory. This reports every
   original file, its readable segment/character counts, original byte hash and
   the exact registered snapshot file hash. It does not depend on the capped
   `/repository/materials/{id}/extracted` preview. Unassociated segments and
   materials without uploaded files are explicitly reported as warnings.
4. Freezes the full inventory before any graph POSTs. It creates one graph run
   per eligible file, avoiding the API's 20-file-per-run limit and keeping a
   readable attachment independent of an unextracted attachment. The four
   supplied Watanabe papers are prioritized, followed by the remaining files in
   ascending character count. No document is dropped because it is long.
5. Computes an identity from material/file IDs, source snapshot hash, material
   title, ontology/config hash, candidate adapter, execution identity and pinned
   pipeline digest. Creation uses a deterministic `Idempotency-Key`. Existing
   graphs are reused only after downloading their immutable `inputs.json` and
   comparing the exact source representation and effective extraction method.
   A listing hint such as `has_saved_candidates` is insufficient.

The stored `snapshot_sha256` hashes the **registered extraction representation**,
which includes its original-file hash and extraction provenance. It is distinct
from `original_sha256`, the uploaded original's byte hash. A changed extraction
record or transcription can therefore produce a new snapshot even if the
original PDF has not changed. A change between inventory and run creation is
detected when the resulting immutable run inputs are checked; that file is
reported as an error and requires a fresh inventory, rather than being mislabeled
as the planned source/method.

Polling reads the complete `primary_batches.json` batch identity list and all
checkpoint **names** from run details. There is no 100-batch cap. Coverage keeps
separate counts for validated batches, rejected batches and untouched batches.
Rejected batches can retain independently validated candidate records, but are
never counted as fully validated. `awaiting_review` or `released` is counted as
complete only when every expected batch has its validated `processed/` checkpoint.
The CLI does not create the `released` state itself.

## Stops, scheduling and recovery

- Default client concurrency is **one**; `--concurrency 2` is the maximum. This
  bounds API submission/monitoring work, not the number of consumers on the
  existing hosted backend or GPU. The CLI changes neither scheduler nor model
  concurrency settings.
- A normal per-pass batch limit resumes only if untouched passages remain and
  checkpoint coverage advances. Rejected checkpoints are preserved rather than
  retried automatically. Provider format failures, other terminal failures,
  unreadable inputs and exhausted validated coverage report `needs_attention`.
- `--timeout-seconds` defaults to one day **per file**, not for the whole library;
  `--max-passes` defaults to 10,000 additional passes per invocation. These bound
  client monitoring. Reaching either limit records an explicit resumable action
  and retains the run ID. Existing hosted jobs are not canceled.
- Repeated unchanged coverage stops a resume loop. A failed file does not prevent
  other planned files from being processed. API/protocol errors retain safe
  machine codes; raw response bodies and source passages are not echoed.
- Ctrl+C preserves completed journal writes and stops further client submissions.
  An in-flight HTTP call can take up to its 60-second request timeout to return.
  An accepted hosted job survives client interruption. Reuse `--resume` afterward.
- One process may own an output directory at a time, enforced by an advisory file
  lock. Use a new directory for another API origin. Do not run multiple backfill
  processes against the same library merely to increase throughput.

Exit code 0 means the requested plan/queue/monitor operation ended without a
recorded error or attention stop. **It does not imply all jobs finished**: inspect
`queued_for_processing`, timeout/pass-budget actions and per-file coverage.
Exit code 1 reports file errors or attention stops; 2 reports setup/protocol
failure; 130 reports client interruption. A dry-run performs no graph writes.

## Local outputs and audit

`OUTPUT_DIR/manifest.json` contains the current plan, method identities, all
historical per-file entries, current-inventory markers, run IDs, fingerprints,
effective config hashes, batch coverage, safe errors and summary counts.
`OUTPUT_DIR/documents/IDENTITY.json` is the atomic per-file journal. Earlier
identities remain available when a file, model or method changes. Reports include
document metadata such as titles and identifiers, so keep the output directory
within the project's ordinary data-access controls.

No original bytes, extracted passages, candidate quotations, raw provider
responses, provider endpoints or credentials are exported into these reports.
Those remain in the hosted repository/run stores with their existing provenance
and retention. The local reports do not delete or expire themselves.

For each finished candidate graph, inspect the evidence in the Workbench:
original-language mention/quote fidelity, entity types and distinct physical
objects, relation endpoints and direction, dates/places, uncertainty and
cross-document identity proposals. Batch coverage only measures processing
accounting. Use the separate research evaluation protocol before claiming a
quality improvement.

## API contract and offline verification

The CLI uses:

| API | Purpose |
| --- | --- |
| `GET /research-runs/capabilities` | Confirm staged candidate and execution identities, pipeline digest and review requirement |
| `GET /research-runs/ontologies/general` | Obtain the exact current ontology and bound generation identity |
| `GET /repository/materials?limit=500&offset=…` | Enumerate all materials before mutations |
| `GET /research-runs/extraction-inventory/{material_id}` | Read all file eligibility, counts and registered snapshot/original hashes |
| `GET /research-runs?material_id=…&limit=100&offset=…` | Enumerate existing runs without relying on a capped first page |
| `POST /research-runs/from-extraction` | Create an immutable candidate run using one registered file and current config |
| `GET /research-runs/{run_id}` | Poll status and the full indexed artifact list |
| `GET /research-runs/{run_id}/artifacts/{name}` | Verify immutable inputs and read the complete batch identity list |
| `POST /research-runs/{run_id}/resume` | Resume deferred untouched passages using the saved immutable configuration |

The API base is supplied explicitly and must have no embedded credentials, query
or fragment. Redirects are rejected. No environment files or credential stores
are read. No dependency, database schema, authentication or deployment changes
are part of this CLI.

Run the focused tests from the backend directory:

```sh
.venv/bin/python -m unittest test.test_reprocess_document_graphs -q
```

The tests mock urllib's HTTP boundary, including request serialization. They cover
pagination beyond 500 materials/100 runs, read-only planning, per-file eligibility,
source-change invalidation, provider identity gates, progress/resume, rejected and
failed batches, active-job timeout, durable queue-only status, source/error-body
redaction and paper prioritization. They execute no provider requests or network
calls. They do not substitute for a deployed canary and source-level review.
