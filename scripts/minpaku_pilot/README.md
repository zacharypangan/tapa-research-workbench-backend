# Minpaku inference pilot

Approved scope: a separate user-owned Ollama installation and only `qwen3:8b`,
`qwen3-vl:8b-instruct`, `qwen3-embedding:0.6b`, with a **25 GB total project disk budget**.
The separately approved worker bridge connects Railway staging only. Production
model routing is unchanged. No sudo, accounts, SSH/VPN or shared Conda changes.
Dedicated staging credentials are held privately, outside this repository.

`release.json` pins the official Ollama **0.40.6** Linux x86_64 artifact, its exact
size and SHA-256 from the official GitHub release (checked 21 October 2026).
The complete `bin/` and `lib/` distribution is retained. NVIDIA RTX 5090 support
is documented by Ollama, but actual driver/backend operation still must pass on
this workstation. Mutable model names are resolved once and recorded with their
full registry digests in each job's inventory; record this inventory with results.

## Transfer and preflight

Use the already authorized personal SSH connection. Do not replace its VPN proxy
or key. If its existing encrypted key is locked, unlock it through the normal
SSH agent or use the user's temporary authorized multiplex connection.

Prepare this layout on the workstation (commands run as the existing user):

```text
/home/zachary/workbench-inference/             # chmod 0700; all pilot data here
  scripts/check_local_model_services.py       # copy from backend/scripts/
  scripts/minpaku_pilot/                      # copy this directory
  runtime/0.40.6/                             # verified installer creates
  models/                                    # isolated registry cache
  state-home/                                # isolated runtime state
  logs/                                      # private logs, manifests and checks
```

Preflight is read-only: inspect Slurm `gpu` availability, `df -h`, `free -h`, the
project path/owner, and `/usr/lib/wsl/lib/nvidia-smi`. The existing personal Python
is `/home/zachary/.conda/envs/zac-research/bin/python`. The checked workstation has
an existing `zstd` executable beside it; the installer also looks there without
modifying that environment. If either prerequisite is absent, stop and identify
the missing prerequisite. The installer does not install dependencies.

## Install, allocate and evaluate

After transfer and project permission checks:

```sh
/home/zachary/.conda/envs/zac-research/bin/python \
  /home/zachary/workbench-inference/scripts/minpaku_pilot/install_runtime.py \
  --project /home/zachary/workbench-inference

sbatch --output=/home/zachary/workbench-inference/logs/slurm-%j.out \
  /home/zachary/workbench-inference/scripts/minpaku_pilot/pilot.sbatch \
  /home/zachary/workbench-inference/scripts/minpaku_pilot \
  /home/zachary/workbench-inference provision
```

Create `logs/` before `sbatch`: Slurm opens its output before running the script.
Installation verifies the pinned hash before extraction, rejects escaping archive
members/links, bounds decompression, and removes temporary archives even on normal
exceptions. It requires at least 25 GB free. It never overwrites another runtime.
If killed with SIGKILL/power loss, inspect any `.install-*` directory before deleting
it and rerunning; do not delete source evidence or model caches to hide a failure.

The provision job queues normally in the existing `gpu` partition with `gpu:1`,
4 CPUs, 16 GB RAM and a **two-hour maximum**, matching the personal notebook
allocation. The whole server, pulls and checks belong to this job; Slurm kills
them at allocation end. The launcher also handles interruption and terminates its
own process groups. It preserves Slurm's `CUDA_VISIBLE_DEVICES` and refuses to run
without an allocation. Never invoke the server binary directly on a login shell.

Ollama listens only at `127.0.0.1:11437` and refuses to start if occupied. It uses
one loaded model, one concurrent inference, queue length 2, context 4096, unloads
between requests and disables Cloud. The subprocess environment contains no
Workbench/Cloud credentials. Loopback limits network exposure; it is not an
authentication boundary against other workstation users. No public endpoint or
shared-browser access is created.

The three active tags are pulled. The original `qwen3-vl:8b` thinking cache is
retained for reproducibility; the explicit instruction variant avoids thinking-only
truncation observed on a real paper. Both fit within the approved total budget. A storage watchdog includes incomplete
blobs, runtime/state and logs, stops at **24 GB** with a 1 GB write safety margin,
and retains partial downloads for inspection/resume. This is admission control,
not an administrator filesystem quota; do not perform unrelated writes in this
directory while the pilot runs. After a successful pull, temporary archives no
longer exist. A failed job does not start an unbounded retry or delete useful data.

The standalone checker evaluates generation, schema-shaped exact-quote extraction,
Chinese/Japanese translation terminology, finite consistent embedding vectors and
a synthetic vision fixture independently. Each service can fail separately. Results
and full model digests are saved under `logs/`; no research text is used or printed.
Passing is a smoke test, not acceptance of research quality or hosted integration.

If downloads exceed two hours, repeat the same `provision` job: Ollama reuses its
isolated cache. To recheck present models use the same `sbatch` command with `check`.
For an allocated, bounded loopback endpoint use `serve`; it requires all three
models already present and ends with the allocation. Observe/cancel with:

```sh
squeue -u zachary
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,AllocTRES
scancel JOBID
```

## Acceptance and next boundary

Record Slurm accounting, installed runtime identity, full model digests, disk
use, synthetic per-service results, and observed GPU/backend use before claiming
success. The launcher makes a synthetic one-token request and requires nonzero
`size_vram` for the loaded model through `/api/ps`; CPU fallback fails the pilot.
This verifies observed GPU use, not peak VRAM or throughput. Confirm no Ollama
child remains after a cancelled job, then repeat a
small check to prove restart recovery. Measure latency/VRAM and run held-out text
and figures from the Watanabe papers, with human-labelled expected evidence.

Railway cannot reach this private loopback listener through this Mac's VPN. An
authenticated outbound leased-task worker is a separate implementation and
deployment decision. Do not expose Ollama, copy a live SQLite database or leave
a laptop tunnel as the production route. Preserve current graph/translation
versions, and use a distinct model/policy identity when creating embedding indexes.

## Authenticated outbound staging worker

The separately approved staging integration uses `worker.py`: Minpaku makes
outbound HTTPS requests to the fixed Railway **staging** API. Ollama remains
loopback-only and is never exposed through a public tunnel. The implementation
does not create production credentials or change the production provider.

The operator creates this private file at
`/home/zachary/workbench-inference/worker-staging.json`, owned by the existing
user with **mode 0600**, using a dedicated staging worker token. Never put the
actual token in Git, a command argument, a report, a screenshot or a log:

```json
{
  "backend_base_url": "https://tapa-research-workbench-backend-graph-staging.up.railway.app/api/v1",
  "worker_id": "minpaku-rtx5090-pilot",
  "token": "REPLACE_WITH_PRIVATE_STAGING_WORKER_TOKEN"
}
```

After staging endpoints and credentials are deployed, end the previous pilot
allocation before queuing this job. The fourth argument is a file path, never
the token itself:

```sh
sbatch --output=/home/zachary/workbench-inference/logs/slurm-%j.out \
  /home/zachary/workbench-inference/scripts/minpaku_pilot/pilot.sbatch \
  /home/zachary/workbench-inference/scripts/minpaku_pilot \
  /home/zachary/workbench-inference worker \
  /home/zachary/workbench-inference/worker-staging.json
```

The launcher retains GPU proof, one-model admission, disk monitoring and the
two-hour lifetime. A clean subprocess environment forwards no Cloud credentials.
The worker accepts only the four Ollama operations `chat`, `generate`, `embed`
and `embeddings`, and the three approved installed models. It never accepts an
arbitrary URL, filesystem path, command, model pull or runtime option. Requests
must carry a canonical SHA-256 of `{"operation":operation,"body":body}`;
the exact body is verified and never rewritten after hashing. Explicit limits
cover context, output, model retention, JSON size and inference duration.

Each claim advertises actual installed model digests. A five-second heartbeat
renews the short task lease during model execution. Results carry the same task,
worker, nonce and request hash; a lost lease discards the result. The backend
validates the operation-specific result before completion. Legacy `/embeddings`
replies omit their model name; any request-based attribution is explicitly
recorded as `response_model_inferred` together with the installed digest.
Errors contain safe categories; worker logs contain only task IDs and states.
Input/output bodies and tokens are not logged or persisted by this worker.

Run focused checks locally without a network or credentials:

```sh
.venv/bin/python -m unittest test.test_minpaku_worker
```

This is a scheduled pilot, not an always-on availability promise. When the
allocation expires or the workstation/VPN fails, stale workers must not appear
available; the broker expires leases and preserves retryable tasks. Verify the
browser → staging API → worker → GPU → validated result path on actual documents
before enabling any automatic backfill or production routing.

## One real-document evaluation from the local Workbench checkout

`evaluate_document.py` accepts a private JSON bundle with `units` containing one
registered unit of at most 1500 characters. Include its actual `source_id`,
`source_version_id`, `unit_id`, `text`, `text_sha256` and page/segment `locator`.
It verifies the hash and does not silently truncate source text. An optional
`ontology` object overrides the default general ontology; the existing generation
guidance is applied. Additional source/image provenance is retained in the audit.
An excerpt must have its own explicit parent-unit/offset association and content
hash; do not present an altered excerpt as an unchanged complete registered unit.

With an already authorized SSH loopback forward to the allocated runtime, run
from the **local backend**, using its existing environment:

```sh
.venv/bin/python scripts/minpaku_pilot/evaluate_document.py \
  --base-url http://127.0.0.1:11437 \
  --input /private/path/source-unit.json \
  --output /private/path/evaluation-new.json \
  --image /private/path/source-figure.png
```

The image is optional. Each request has a 90-second timeout; extraction defaults
to one attempt, with `--attempts 2` allowing bounded repair. Context is 12,288
tokens (optionally 16,384) because the complete ontology/evidence prompt itself
exceeds the simple fixture's context. Output is capped at 4096 tokens, translation
at 2048, and image labels at 4096; truncated responses fail. Only a loopback HTTP
origin is accepted and environment proxy settings are ignored.

The unchanged production prompt/schema and exact-span validator process graph
candidates. Short request-local unit IDs map back exactly; no fuzzy quotations,
invented evidence or validator relaxation. The private output records raw attempts,
accepted/rejected counts and reasons, model digests, source/ontology/request hashes,
durations, a separate English copy, embedding checks and optional observable image
labels. Its file mode is 0600 and it refuses overwriting an earlier artifact.
No database, index, graph release or translated source record is changed. Operational
success, candidate count and scientific correctness remain separate assessments.

Sources: [official release](https://github.com/ollama/ollama/releases/tag/v0.40.6),
[Linux manual distribution](https://docs.ollama.com/linux),
[hardware support](https://docs.ollama.com/gpu),
[Ollama controls](https://docs.ollama.com/faq),
[Slurm GPU allocation](https://slurm.schedmd.com/gres.html).

## Gateway contract and rollback

Railway exposes `/api/v1/model-gateway/api/{chat,generate,embed,embeddings}`
for the existing Ollama adapter and `/api/v1/model-workers/{claim,heartbeat,
result,failure,status}` for the private worker. `status` uses the gateway token;
other worker routes use the independent worker token. Both secrets must have at
least 32 characters and differ; missing/equal secrets disable the routes. Never
configure them in production as part of this pilot.

The gateway preserves a hash of the requested body, then applies versioned
`minpaku-bounded-v1` admission: context 16,384, thinking off, no retained model,
non-streamed output <=4096 tokens, graph output <=6 entities/6 assertions and
<=2 registered span references per candidate. Embedding truncation is disabled.
Effective request hashes and limits accompany successful responses. The original
graph validator remains unchanged. Graph counts are per request, not document
coverage limits. Incomplete outputs are rejected. OCR labels must preserve exact
source spacing/glyphs; English names belong in derived labels.

Only bounded transient task envelopes occupy the existing Railway storage: 32
pending requests, 64 MiB total, 8 MiB per request/result, 20-second leases, three
attempts, deadlines <=80 seconds. Terminal inputs are erased immediately and
terminal envelopes expire after five minutes. Original documents and released
graphs are outside this cleanup. Archive effective-policy/model audits with
research evaluation artifacts before transient cleanup; the existing pinned
extraction adapter does not automatically retain all additional gateway metadata.

Enable in this order: deploy disabled-by-default code; set distinct staging
`MODEL_GATEWAY_TOKEN`/`MODEL_WORKER_TOKEN`; start the Slurm worker with its private
configuration; verify recent worker/model digests; then set staging-only
`REPOSITORY_OLLAMA_BASE_URL` to the staging origin plus
`/api/v1/model-gateway`, `REPOSITORY_OLLAMA_API_KEY` to the gateway token,
`REPOSITORY_EMBEDDING_PROVIDER=ollama`, and the three `OLLAMA_*_MODEL` values to
the active release tags. Preserve previous values privately for rollback.

Rollback restores those staging provider values and redeploys, then cancels the
worker allocation. It does not delete source files, graph releases or model
caches. The allocation ends after two hours; continuous service requires an
administrator-approved recurring reservation and worker supervision. A laptop
SSH forward is evaluation-only and is not the hosted service transport.
