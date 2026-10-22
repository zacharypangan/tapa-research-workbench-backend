#!/usr/bin/env python3
"""Plan or resume exhaustive, unreviewed document graph backfills via hosted APIs.

Standard-library only. No source passages, provider responses or credentials are
written to the local manifest. Execute submits normal Workbench graph jobs; it
never accepts, publishes, deletes or collects additional internet sources.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


MANIFEST_VERSION = "document-graph-backfill/1"
REQUIRED_METHOD = "staged_document_claims"
ACTIVE = {"created", "queued", "running"}
READY = {"awaiting_review", "released"}
PRIORITY_PAPERS = ("台湾地区树皮布石拍初探", "广西发现的史前石拍初探",
                   "广西与越南发现的早期石拍研究", "史前蒙古人种海洋扩散研究")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


class BackfillError(RuntimeError):
    """Safe error code: never expose HTTP bodies, passages or credential values."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    def __init__(self, base):
        self.base = base.rstrip("/")
        parts = urlsplit(self.base)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username or parts.password or parts.query or parts.fragment):
            raise BackfillError("invalid_api_base")
        self.opener = build_opener(NoRedirect())

    def request(self, path, body=None, key=None):
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if key:
            headers["Idempotency-Key"] = key
        data = None if body is None else json.dumps(body).encode()
        try:
            with self.opener.open(Request(self.base + path, data=data, headers=headers), timeout=60) as response:
                raw = response.read(128 * 1024 * 1024 + 1)
                if len(raw) > 128 * 1024 * 1024:
                    raise BackfillError("response_size_limit")
                return json.loads(raw)
        except HTTPError as exc:
            raise BackfillError(f"http_{exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise BackfillError("request_unavailable") from None
        except (ValueError, UnicodeError):
            raise BackfillError("invalid_json_response") from None


def pages(client, path, key, limit):
    """Freeze a complete listing before writes; fail rather than truncate."""
    rows, offset, expected = [], 0, None
    while True:
        response = client.request(path + ("&" if "?" in path else "?") + urlencode({"limit": limit, "offset": offset}))
        page, total = response.get(key), response.get("total")
        if not isinstance(page, list) or type(total) is not int or total < 0:
            raise BackfillError("invalid_pagination")
        if expected is not None and total != expected:
            raise BackfillError("listing_changed_during_pagination")
        expected = total
        rows.extend(page)
        offset += len(page)
        if offset >= total:
            if offset != total:
                raise BackfillError("invalid_pagination")
            return rows
        if not page or (path.startswith("/research-runs") and offset > 100000):
            raise BackfillError("incomplete_pagination")


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def method_config(client, batch_chars, max_auto_batches):
    capability = client.request("/research-runs/capabilities")
    extraction = capability.get("extraction", {})
    provider = extraction.get("candidate_provider", {})
    execution = extraction.get("execution_identity", {})
    pipeline_hash = capability.get("pipeline_implementation_sha256")
    # Permit the hash to be nested under extraction for a compatible deployment.
    pipeline_hash = pipeline_hash or extraction.get("pipeline_implementation_sha256")
    version = provider.get("adapter_version", provider.get("version"))
    if (not capability.get("registered_extraction", {}).get("available")
            or extraction.get("configured") is not True
            or provider.get("method") != REQUIRED_METHOD or provider.get("requires_review") is not True
            or not isinstance(version, str) or not version or not sha(provider.get("code_sha256"))
            or not sha(pipeline_hash) or not isinstance(execution.get("model"), str) or not execution["model"]
            or not sha(execution.get("provider_sha256")) or not sha(execution.get("runtime_code_sha256"))):
        raise BackfillError("improved_provider_not_deployed_or_configured")
    ontology = client.request("/research-runs/ontologies/general")
    guidance = ontology.get("extraction_guidance", {})
    bound = guidance.get("candidate_provider", {})
    if (bound.get("method") != REQUIRED_METHOD or bound.get("code_sha256") != provider["code_sha256"]
            or bound.get("adapter_version", bound.get("version")) != version
            or guidance.get("execution_identity") != execution):
        raise BackfillError("provider_identity_not_bound_to_ontology")
    # Explicit fields participate in identity; disabled collection is deliberate.
    config = {"ontology": ontology, "research_topic": "documented research evidence",
              "processing": {"mode": "ollama", "batch_chars": batch_chars, "max_auto_batches": max_auto_batches},
              "augmentation": {"enabled": False}}
    identity = {"protocol": MANIFEST_VERSION, "candidate_provider_version": version,
                "candidate_provider_code_sha256": provider["code_sha256"],
                "execution_identity": execution, "pipeline_implementation_sha256": pipeline_hash,
                "requested_config_sha256": digest(config)}
    return config, identity


def artifact(client, run_id, name):
    return client.request(f"/research-runs/{quote(run_id, safe='')}/artifacts/{quote(name, safe='/')}")


def matches(inputs, row, config, method):
    """Only identical registered source + effective method can be reused."""
    documents = inputs.get("documents", [])
    actual = inputs.get("config", {})
    return (len(documents) == 1 and documents[0].get("original_locator") == row["source_locator"]
            and documents[0].get("sha256") == row["snapshot_sha256"]
            and documents[0].get("representation_kind") == "registered_extraction"
            and documents[0].get("title") == row["material_title"]
            and digest(inputs.get("implementation_sha256", {})) == method["pipeline_implementation_sha256"]
            and actual.get("ontology") == config["ontology"]
            and actual.get("processing") == config["processing"]
            and actual.get("research_topic") == config["research_topic"]
            and actual.get("augmentation", {}).get("enabled") is False)


def coverage(client, run, cached_batches):
    """Account for every batch, including rejected and never attempted batches."""
    names = set(run.get("artifacts", []))
    if "primary_batches.json" not in names:
        return {"total_batches": None, "validated_batches": 0, "rejected_batches": 0,
                "attempted_batches": 0, "unattempted_batches": None, "all_batches_validated": False}
    if run["run_id"] not in cached_batches:
        batches = artifact(client, run["run_id"], "primary_batches.json")
        ids = [batch["batch_id"] for batch in batches]
        if len(set(ids)) != len(ids):
            raise BackfillError("duplicate_batch_identity")
        cached_batches[run["run_id"]] = set(ids)
    ids = cached_batches[run["run_id"]]
    processed = {item for item in ids if f"processed/{item}.json" in names}
    rejected = {item for item in ids if f"rejected_batches/{item}.json" in names} - processed
    return {"total_batches": len(ids), "validated_batches": len(processed), "rejected_batches": len(rejected),
            "attempted_batches": len(processed | rejected), "unattempted_batches": len(ids - processed - rejected),
            "all_batches_validated": bool(ids) and processed == ids}


def safe_error(exc):
    return str(exc) if isinstance(exc, BackfillError) else "unexpected_client_error"


def eligibility_reason(value):
    known = {"Original file is unavailable in managed storage": "original_unavailable",
             "No registered readable extraction; extract this file first": "no_readable_segments",
             "Registered extraction exceeds the 50 MiB graph input limit": "snapshot_size_limit",
             "Original file could not be read; retry after storage is available": "original_read_unavailable"}
    if isinstance(value, str):
        return known.get(value, value if re.fullmatch(r"[a-z_]{1,100}", value) else "needs_readable_extraction")
    return "needs_readable_extraction"


class Backfill:
    def __init__(self, client, args, *, clock=time.monotonic, sleep=time.sleep):
        self.client, self.args, self.clock, self.sleep = client, args, clock, sleep
        self.output = args.output_dir
        self.path = self.output / "manifest.json"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.cached_batches = {}
        self.manifest = {"version": MANIFEST_VERSION, "api_base": client.base,
                         "created_at": now(), "documents": []}
        if self.path.exists():
            if not args.resume:
                raise BackfillError("manifest_exists_use_resume")
            self.manifest = json.loads(self.path.read_text())
            if self.manifest.get("version") != MANIFEST_VERSION or self.manifest.get("api_base") != client.base:
                raise BackfillError("manifest_origin_or_version_mismatch")

    def save(self, row=None):
        with self.lock:
            self.output.mkdir(parents=True, exist_ok=True)
            self.manifest["updated_at"] = now()
            if row:
                document_path = self.output / "documents" / (row["identity"] + ".json")
                document_path.parent.mkdir(exist_ok=True)
                self.atomic(document_path, row)
            self.atomic(self.path, self.manifest)

    @staticmethod
    def atomic(path, value):
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def plan(self):
        self.config, self.method = method_config(self.client, self.args.batch_chars,
                                                None if self.args.all_batches else self.args.max_auto_batches)
        materials = pages(self.client, "/repository/materials", "materials", 500)
        if len({item["id"] for item in materials}) != len(materials):
            raise BackfillError("duplicate_material_listing")
        selected = set(self.args.material_id or [])
        if selected - {item["id"] for item in materials}:
            raise BackfillError("selected_material_not_found")
        materials = [item for item in materials if not selected or item["id"] in selected]
        self.manifest.update({"method": self.method, "executed": self.args.execute,
                              "selected_material_ids": sorted(item["id"] for item in materials),
                              "requires_researcher_review": True, "inventory_warnings": []})
        existing = {row["identity"]: row for row in self.manifest["documents"]}
        tasks = []
        # Complete the source/run inventory before creating any jobs, since the
        # material listing is ordered by mutable updated_at values.
        for material in materials:
            material_id = material["id"]
            inventory = self.client.request("/research-runs/extraction-inventory/" + quote(material_id, safe=""))
            if inventory.get("material_id") != material_id or not isinstance(inventory.get("files"), list):
                raise BackfillError("invalid_extraction_inventory")
            if len({file["file_id"] for file in inventory["files"]}) != len(inventory["files"]):
                raise BackfillError("duplicate_inventory_file")
            if inventory.get("unassociated_segment_count"):
                self.manifest["inventory_warnings"].append({"material_id": material_id, "reason": "unassociated_segments_not_file_evidence",
                                                            "count": inventory["unassociated_segment_count"]})
            if not inventory["files"]:
                self.manifest["inventory_warnings"].append({"material_id": material_id, "reason": "no_registered_files"})
            saved = pages(self.client, "/research-runs?" + urlencode({"material_id": material_id}), "items", 100)
            for file in inventory["files"]:
                file_id = file["file_id"]
                source = {"material_id": material_id, "file_id": file_id,
                          "snapshot_sha256": file.get("snapshot_sha256"), "material_title": material["title"]}
                identity = digest({"source": source, "method": self.method})
                row = existing.get(identity)
                if row is None:
                    row = {**source, "identity": identity, "created_at": now(), "passes_queued": 0}
                    self.manifest["documents"].append(row)
                    existing[identity] = row
                row.update({"current_inventory": True, "eligible": file.get("eligible") is True,
                            "source_locator": f"workbench:material/{material_id}/file/{file_id}",
                            "original_sha256": file.get("original_sha256"),
                            "segment_count": file.get("segment_count", 0), "character_count": file.get("character_count", 0),
                            "extraction_status": eligibility_reason(inventory.get("extraction_status", "not_extracted")),
                            "requires_researcher_review": True})
                if not row["eligible"]:
                    # Server reason is a short machine category, never copy an
                    # arbitrary provider/SQL error body into an export.
                    row.update(action="needs_extraction", status="not_eligible",
                               reason=eligibility_reason(file.get("reason")))
                    self.save(row)
                    continue
                if not sha(file.get("snapshot_sha256")) or not sha(file.get("original_sha256")):
                    raise BackfillError("eligible_file_missing_source_hash")
                row["run_title"] = ("Document graph — " + file.get("original_filename", file_id))[:200]
                filename = file.get("original_filename", "")
                row["priority"] = next((index for index, title in enumerate(PRIORITY_PAPERS) if title in filename), len(PRIORITY_PAPERS))
                candidate_ids = [row["run_id"]] if row.get("run_id") else []
                candidate_ids += [run["run_id"] for run in saved
                                  if run.get("processing_mode") == "ollama" and any(
                                      doc.get("locator") == row["source_locator"] for doc in run.get("documents", []))]
                row.pop("error", None)
                tasks.append((row, list(dict.fromkeys(candidate_ids))))
                self.save(row)
        tasks.sort(key=lambda task: (task[0]["priority"], task[0]["character_count"], task[0]["material_id"], task[0]["file_id"]))
        return tasks

    def process(self, row, candidate_ids):
        try:
            if self.stop.is_set():
                row["action"] = "interrupted_resume_later"
                return
            run = None
            for identifier in candidate_ids:
                inputs = artifact(self.client, identifier, "inputs.json")
                if matches(inputs, row, self.config, self.method):
                    run = self.client.request("/research-runs/" + quote(identifier, safe=""))
                    row["effective_config_sha256"] = digest(inputs["config"])
                    row["run_fingerprint"] = inputs.get("fingerprint")
                    break
            if run is None and not self.args.execute:
                row.update(action="would_create", status="planned")
                return
            if run is None:
                body = {"material_id": row["material_id"], "file_ids": [row["file_id"]],
                        "title": row["run_title"], "config": self.config}
                run = self.client.request("/research-runs/from-extraction", body, "graph-backfill:" + row["identity"])
                row.update(run_id=run["run_id"], action="created", passes_queued=row["passes_queued"] + 1)
                # Persist immediately, then check the server-bound identity.
                self.save(row)
                inputs = artifact(self.client, run["run_id"], "inputs.json")
                if not matches(inputs, row, self.config, self.method):
                    raise BackfillError("source_or_method_changed_during_creation")
                row["effective_config_sha256"] = digest(inputs["config"])
                row["run_fingerprint"] = inputs.get("fingerprint")
            row["run_id"] = run["run_id"]
            if not self.args.execute:
                self.update(row, run)
                row["action"] = "would_skip_complete" if row["complete"] else "would_resume"
                return
            if self.args.queue_only:
                self.update(row, run)
                if row["complete"]:
                    row["action"] = "complete_candidates_for_review"
                elif run["status"] in ACTIVE:
                    if run["status"] == "created":
                        run = self.client.request("/research-runs/" + quote(run["run_id"], safe="") + "/resume", {})
                        row["passes_queued"] += 1
                        self.update(row, run)
                    row["action"] = "queued_for_processing"
                else:
                    row["action"] = "needs_attention"
                return
            self.follow(row, run)
        except Exception as exc:
            row.update(action="error", complete=False, error=safe_error(exc))
        finally:
            self.save(row)

    def update(self, row, run):
        row.update(status=run["status"], coverage=coverage(self.client, run, self.cached_batches), updated_at=now())
        row["complete"] = run["status"] in READY and row["coverage"]["all_batches_validated"]

    def follow(self, row, run):
        deadline, passes, previous_progress = self.clock() + self.args.timeout_seconds, 0, None
        while True:
            self.update(row, run)
            self.save(row)
            state, covered = row["status"], row["coverage"]
            if row["complete"]:
                row["action"] = "complete_candidates_for_review"
                return
            if self.stop.is_set():
                row["action"] = "interrupted_resume_later"
                return
            if state in READY:
                row["action"] = "needs_attention_incomplete_coverage"
                return
            if self.clock() >= deadline:
                row["action"] = "time_budget_reached_resume_later"
                return
            if state not in ACTIVE:
                # Rejected checkpoints are never rerun automatically. Continue
                # deferred, untouched passages only after a normal pass budget.
                resumable = (state == "awaiting_annotations" and run.get("automatic_batch_budget_reached") is True
                             and not run.get("provider_format_budget_reached")
                             and covered["unattempted_batches"] is not None and covered["unattempted_batches"] > 0)
                if not resumable:
                    row["action"] = "needs_attention"
                    return
                progress = (covered["validated_batches"], covered["rejected_batches"])
                if progress == previous_progress:
                    row["action"] = "needs_attention_no_progress"
                    return
                if passes >= self.args.max_passes:
                    row["action"] = "pass_budget_reached_resume_later"
                    return
                previous_progress = progress
                run = self.client.request("/research-runs/" + quote(run["run_id"], safe="") + "/resume", {})
                passes += 1
                row["passes_queued"] += 1
                self.save(row)
            self.sleep(min(self.args.poll_seconds, max(0, deadline - self.clock())))
            run = self.client.request("/research-runs/" + quote(run["run_id"], safe=""))

    def run(self):
        for row in self.manifest["documents"]:
            row["current_inventory"] = False
        tasks = self.plan()
        pool = ThreadPoolExecutor(max_workers=self.args.concurrency)
        try:
            list(pool.map(lambda task: self.process(*task), tasks))
        except KeyboardInterrupt:
            self.stop.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=self.stop.is_set())
        current = [row for row in self.manifest["documents"] if row.get("current_inventory")]
        self.manifest["summary"] = {"files": len(current), "eligible": sum(row["eligible"] for row in current),
                                    "complete_candidates_for_review": sum(row.get("complete", False) for row in current),
                                    "queued_for_processing": sum(row.get("action") == "queued_for_processing" for row in current),
                                    "errors": sum(row.get("action") == "error" for row in current),
                                    "needs_attention": sum(str(row.get("action", "")).startswith("needs_attention") for row in current)}
        self.save()
        return self.manifest["summary"]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True, help="Workbench API base, including /api/v1")
    parser.add_argument("--output-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Read-only plan (default)")
    mode.add_argument("--execute", action="store_true", help="Create/resume candidate graph jobs")
    parser.add_argument("--resume", action="store_true", help="Continue an existing local manifest")
    parser.add_argument("--material-id", action="append", help="Restrict to a canary; repeatable")
    parser.add_argument("--batch-chars", type=int, default=1500)
    parser.add_argument("--max-auto-batches", type=int, default=4, help="Budget per server pass; subsequent passes resume")
    parser.add_argument("--all-batches", action="store_true", help="No per-pass batch cap; hosted jobs process all passages")
    parser.add_argument("--queue-only", action="store_true", help="With --execute --all-batches, durably enqueue then return; completion is not claimed")
    parser.add_argument("--concurrency", type=int, choices=(1, 2), default=1)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=86400, help="Per-file wall-clock budget; jobs survive CLI exit")
    parser.add_argument("--max-passes", type=int, default=10000, help="Maximum additional passes per file per invocation")
    args = parser.parse_args(argv)
    if args.queue_only and (not args.execute or not args.all_batches):
        parser.error("--queue-only requires --execute --all-batches")
    if (not 1000 <= args.batch_chars <= 1500 or not 1 <= args.max_auto_batches <= 100
            or not 0 < args.poll_seconds <= 300 or not 0 < args.timeout_seconds <= 604800
            or not 1 <= args.max_passes <= 100000):
        parser.error("Use batch-chars 1000–1500, batches 1–100, positive poll ≤300s, timeout ≤7 days, passes 1–100000")
    return args


def main(argv=None):
    args = arguments(argv)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / ".backfill.lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise BackfillError("output_directory_in_use") from None
            summary = Backfill(Client(args.api_base), args).run()
        print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), **summary}))
        return 1 if summary["errors"] or summary["needs_attention"] else 0
    except (BackfillError, OSError, ValueError) as exc:
        print(json.dumps({"error": safe_error(exc)}))
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "resume": True}))
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
