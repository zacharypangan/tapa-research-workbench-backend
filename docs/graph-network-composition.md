# Saved networks and the document database

The service combines existing saved candidates without making extraction/provider requests. It stores a derived JSON snapshot under the existing persistent storage root (`graph_networks/`). Existing runs, their input hashes, approvals and published releases stay intact. A combined network is a draft requiring review.

## Replicable calls

Use the same API base as the Workbench (paths below are relative to `/api/v1`).

1. List available saved research runs using `GET /research-runs?limit=100&offset=0`; read every page. Inspect a run's extraction coverage rather than treating saved candidates as full extraction.
2. `POST /graph-networks` with `{"title":"Paper comparison","run_ids":["RUN_UUID_1","RUN_UUID_2"],"scope":"selected","infer_connections":true}`. One run is also valid: this creates an enriched draft of that run.
3. `GET /graph-networks/{network_id}/graph` returns the standard graph plus separate `inferences`, original-run provenance and original review decisions. Reported assertions, analytical alignments and derived inferences remain separate.
4. `GET /graph-networks/{network_id}/export` downloads the network metadata, coverage register and graph as JSON.
5. For an entire-library document database snapshot, `POST /graph-networks` with `{"title":"Document database","scope":"library","infer_connections":true}`. Library scope rejects manually supplied run IDs. It reads all material records, all run pages and every safe candidate checkpoint, selects the most populated current graph per document and includes documents lacking a graph in the register.

`GET /graph-networks?limit=20&offset=0` lists snapshots. Repeating the same semantic inputs creates the same stable network ID; processing more source passages or changing selection produces a new snapshot. Snapshots do not update automatically.

## What is preserved

All graph identifiers and identity reference fields receive the full original run UUID namespace, including nested scope/seed bindings, source versions, unit IDs and evidence/entity version-map keys. Global Workbench material/file IDs, source locators, literal attribute values, quotation text, hashes and Unicode quotation offsets remain intact. Each record carries its original run and record ID. Sources still open their original Workbench material records.

Original input hashes and current registered file hashes are verified before composition. Deleted, replaced or modified originals cannot silently contribute a new network. Library mode records excluded runs and reasons; selected mode rejects the request. Independently validated candidates from partially rejected batches remain partial candidates; rejected batch outputs are never accepted as complete checkpoints.

A published run contributes its authoritative release only. Draft checkpoints cannot add omitted candidates to that release, and a missing release cannot silently fall back to drafts. Draft runs can still combine every independently validated checkpoint.

Origin review history is retained only as provenance. Active combined decisions are empty; upstream approvals do not certify this new draft or inferred edges. The latest valid original rejection is retained for inference eligibility. No physical entity identity merge, accepted historical transmission, semantic fact transfer or scientific publication occurs.

## Coverage and limits

The database register includes every current material and its attached files, extraction segment counts, graph participation and pending batch counts. A file contributes graph coverage only when it has at least one actual concept, not merely because it was attached to a multi-file run. Missing text, missing graphs, unprocessed passages and partial file coverage remain visible. The database is a combined saved graph and document register, not a promise of exhaustive extraction or validated scientific content.

Selected composition accepts 1–200 unique run IDs. Library scope has no 200-document truncation. Lists paginate; graph loading does not use the 20-checkpoint listing hint as a completeness boundary. Inference has its own explicit connection limits and report; a capped proposal list does not imply every possible connection was considered.

Library selection chooses the most populated current saved run per document. If a chosen run contains several documents, its full source-preserving graph can bring an additional saved account for another document. Those source-local accounts keep distinct IDs; the register reports `saved_account_count` and explains this case rather than silently merging them.

New snapshots use the existing storage reserve, atomic checkpoint save and a per-network lock. Concurrent identical requests reuse one snapshot. A failed write cleans up its temporary file and returns a sanitized HTTP 503; unreadable cached networks are excluded from listings with `unavailable_count`, and direct reads/exports return 503 without filesystem details.

Run the offline checks with `.venv/bin/python -m unittest test.test_graph_networks -q`. No new dependencies, database schema, authentication or deployment configuration are required.
