"""Offline HTTP-contract tests: no server, provider or network is used."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit


SPEC = importlib.util.spec_from_file_location("reprocess_document_graphs", Path(__file__).parents[1] / "scripts/reprocess_document_graphs.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class FakeResponse(io.BytesIO):
    def __init__(self, value):
        super().__init__(json.dumps(value).encode())


class Clock:
    def __init__(self):
        self.value = 0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class HTTPFixture:
    """Mock urllib at its HTTP boundary, retaining actual client serialization."""
    def __init__(self):
        self.calls, self.runs, self.saved = [], {}, []
        self.implementation = {"store.py": "a" * 64}
        self.provider = {"method": "staged_document_claims", "version": "staged-test/1", "code_sha256": "b" * 64,
                         "requires_review": True}
        self.execution = {"model": "fixture-model", "provider_sha256": "c" * 64, "runtime_code_sha256": "d" * 64}
        self.ontology = {"version": "general/1", "entity_types": {"Concept": "Concept"}, "predicates": {},
                         "extraction_guidance": {"candidate_provider": copy.deepcopy(self.provider),
                                                 "execution_identity": copy.deepcopy(self.execution)}}
        self.materials = [{"id": "material1", "title": "Paper", "segment_count": 1}]
        self.files = [{"file_id": "file1", "eligible": True, "snapshot_sha256": "e" * 64,
                       "original_sha256": "f" * 64, "segment_count": 2, "character_count": 2800,
                       "original_filename": "Paper.pdf", "reason": None}]
        self.mode = "resume"
        self.batch_requests = 0

    def open(self, request, timeout):
        parts = urlsplit(request.full_url)
        path = parts.path.removeprefix("/api/v1")
        query = parse_qs(parts.query)
        body = json.loads(request.data) if request.data is not None else None
        self.calls.append((request.get_method(), path, query, body, dict(request.header_items())))
        if path == "/research-runs/capabilities":
            return FakeResponse({"registered_extraction": {"available": True},
                                 "pipeline_implementation_sha256": cli.digest(self.implementation),
                                 "extraction": {"configured": True, "candidate_provider": self.provider,
                                                "execution_identity": self.execution}})
        if path == "/research-runs/ontologies/general":
            return FakeResponse(self.ontology)
        if path == "/repository/materials":
            return self.listing(self.materials, "materials", query)
        if path.startswith("/research-runs/extraction-inventory/"):
            return FakeResponse({"material_id": path.rsplit("/", 1)[1], "files": self.files,
                                 "unassociated_segment_count": 0})
        if path == "/research-runs" and body is None:
            return self.listing(self.saved, "items", query)
        if path == "/research-runs/from-extraction":
            identifier = "run" + str(len(self.runs) + 1)
            file = next(row for row in self.files if row["file_id"] == body["file_ids"][0])
            inputs = {"documents": [{"original_locator": f"workbench:material/{body['material_id']}/file/{file['file_id']}",
                                     "sha256": file["snapshot_sha256"], "representation_kind": "registered_extraction", "title": "Paper"}],
                      "config": body["config"], "implementation_sha256": self.implementation, "fingerprint": "1" * 64}
            self.runs[identifier] = {"inputs": inputs, "phase": "initial"}
            return FakeResponse(self.description(identifier))
        if path.endswith("/resume"):
            identifier = path.split("/")[2]
            self.runs[identifier]["phase"] = "resumed"
            return FakeResponse(self.description(identifier, queued=True))
        if "/artifacts/" in path:
            identifier, name = path.split("/")[2], path.split("/artifacts/", 1)[1]
            if name == "inputs.json":
                return FakeResponse(self.runs[identifier]["inputs"])
            if name == "primary_batches.json":
                self.batch_requests += 1
                return FakeResponse([{"batch_id": "B1", "units": [{"text": "private source text"}]}, {"batch_id": "B2", "units": []}])
        if path.startswith("/research-runs/"):
            return FakeResponse(self.description(path.split("/")[2]))
        raise AssertionError((request.get_method(), path))

    @staticmethod
    def listing(rows, key, query):
        offset, limit = int(query["offset"][0]), int(query["limit"][0])
        return FakeResponse({key: rows[offset:offset + limit], "total": len(rows)})

    def description(self, identifier, queued=False):
        phase = self.runs[identifier]["phase"]
        names = ["inputs.json", "primary_batches.json", "processed/B1.json"]
        if self.mode == "active" or queued:
            return {"run_id": identifier, "status": "queued", "artifacts": names}
        if phase == "resumed" or self.mode == "complete":
            return {"run_id": identifier, "status": "awaiting_review", "artifacts": names + ["processed/B2.json"]}
        if self.mode == "rejected":
            return {"run_id": identifier, "status": "awaiting_annotations", "artifacts": names + ["rejected_batches/B2.json"],
                    "automatic_batch_budget_reached": False}
        if self.mode == "failed":
            return {"run_id": identifier, "status": "failed", "artifacts": names,
                    "detail": "SECRET provider body and private source passage"}
        if self.mode == "stalled":
            return {"run_id": identifier, "status": "awaiting_annotations", "artifacts": names,
                    "automatic_batch_budget_reached": True}
        return {"run_id": identifier, "status": "awaiting_annotations", "artifacts": names,
                "automatic_batch_budget_reached": True}


class ReprocessDocumentGraphsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = HTTPFixture()
        self.client = cli.Client("https://example.test/api/v1")
        self.client.opener = self.fixture
        self.clock = Clock()

    def args(self, *extra):
        return cli.arguments(["--api-base", self.client.base, "--output-dir", self.directory.name,
                              "--poll-seconds", "1", "--timeout-seconds", "5", *extra])

    def run_backfill(self, *extra):
        run = cli.Backfill(self.client, self.args(*extra), clock=self.clock.monotonic, sleep=self.clock.sleep)
        run.run()
        return run

    def test_dry_run_is_read_only_and_reports_all_files_independently(self):
        self.fixture.files.append({"file_id": "file2", "eligible": False, "reason": "no_readable_segments", "segment_count": 0})
        run = self.run_backfill("--dry-run")
        self.assertEqual(run.manifest["summary"]["files"], 2)
        self.assertEqual([row["action"] for row in run.manifest["documents"]], ["would_create", "needs_extraction"])
        self.assertTrue(all(call[0] == "GET" for call in self.fixture.calls))

    def test_full_material_and_run_pagination_not_capped(self):
        self.fixture.materials = [{"id": str(index)} for index in range(501)]
        rows = cli.pages(self.client, "/repository/materials", "materials", 500)
        self.assertEqual(len(rows), 501)
        self.fixture.saved = [{"run_id": str(index)} for index in range(101)]
        rows = cli.pages(self.client, "/research-runs?material_id=material1", "items", 100)
        self.assertEqual(len(rows), 101)
        self.assertEqual(self.fixture.calls[-1][2]["offset"], ["100"])

    def test_budgeted_pass_resumes_every_batch_and_does_not_save_source_text(self):
        run = self.run_backfill("--execute")
        row = run.manifest["documents"][0]
        self.assertEqual(row["coverage"], {"total_batches": 2, "validated_batches": 2, "rejected_batches": 0,
                                           "attempted_batches": 2, "unattempted_batches": 0, "all_batches_validated": True})
        self.assertTrue(row["complete"])
        self.assertEqual(row["passes_queued"], 2)
        self.assertEqual(self.fixture.batch_requests, 1)
        self.assertEqual(len([call for call in self.fixture.calls if call[1].endswith("/resume")]), 1)
        self.assertNotIn("private source text", run.path.read_text())
        self.assertTrue(list((Path(self.directory.name) / "documents").glob("*.json")))

    def test_resume_reuses_unchanged_complete_run_with_no_post(self):
        self.run_backfill("--execute")
        self.fixture.calls.clear()
        run = self.run_backfill("--execute", "--resume")
        self.assertTrue(run.manifest["documents"][0]["complete"])
        self.assertFalse(any(call[0] == "POST" for call in self.fixture.calls))

    def test_changed_source_creates_new_identity_instead_of_skipping_old_graph(self):
        first = self.run_backfill("--execute")
        self.fixture.files[0]["snapshot_sha256"] = "9" * 64
        self.fixture.calls.clear()
        second = self.run_backfill("--execute", "--resume")
        self.assertEqual(len(second.manifest["documents"]), 2)
        current = [row for row in second.manifest["documents"] if row["current_inventory"]]
        self.assertEqual(len(current), 1)
        self.assertNotEqual(current[0]["identity"], first.manifest["documents"][0]["identity"])
        self.assertEqual(len([call for call in self.fixture.calls if call[1].endswith("from-extraction")]), 1)

    def test_all_batches_queue_only_is_durable_queue_not_completion(self):
        self.fixture.mode = "active"
        run = self.run_backfill("--execute", "--all-batches", "--queue-only")
        create = next(call for call in self.fixture.calls if call[1].endswith("from-extraction"))
        self.assertIsNone(create[3]["config"]["processing"]["max_auto_batches"])
        self.assertEqual(create[3]["config"]["augmentation"], {"enabled": False})
        self.assertEqual(run.manifest["documents"][0]["action"], "queued_for_processing")
        self.assertFalse(run.manifest["documents"][0]["complete"])
        self.assertEqual(run.manifest["summary"]["queued_for_processing"], 1)

    def test_every_attachment_beyond_twenty_is_queued_independently(self):
        self.fixture.mode = "active"
        self.fixture.files = [{**self.fixture.files[0], "file_id": f"file{index}"} for index in range(21)]
        run = self.run_backfill("--execute", "--all-batches", "--queue-only")
        creates = [call for call in self.fixture.calls if call[1].endswith("from-extraction")]
        self.assertEqual(len(creates), 21)
        self.assertTrue(all(len(call[3]["file_ids"]) == 1 for call in creates))
        self.assertEqual(run.manifest["summary"]["queued_for_processing"], 21)

    def test_matching_completion_after_first_hundred_runs_is_reused(self):
        config, _ = cli.method_config(self.client, 1500, 4)
        self.fixture.mode = "complete"
        self.fixture.runs["existing"] = {"phase": "initial", "inputs": {
            "documents": [{"original_locator": "workbench:material/material1/file/file1", "sha256": "e" * 64,
                           "representation_kind": "registered_extraction", "title": "Paper"}],
            "config": config, "implementation_sha256": self.fixture.implementation}}
        self.fixture.saved = [{"run_id": f"old{index}", "processing_mode": "manual"} for index in range(100)] + [
            {"run_id": "existing", "processing_mode": "ollama", "documents": [{"locator": "workbench:material/material1/file/file1"}]}]
        run = self.run_backfill("--execute")
        self.assertEqual(run.manifest["documents"][0]["run_id"], "existing")
        self.assertTrue(run.manifest["documents"][0]["complete"])
        self.assertFalse(any(call[0] == "POST" for call in self.fixture.calls))

    def test_rejected_and_failed_files_are_not_marked_complete_or_silently_retried(self):
        for mode in ("rejected", "failed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                self.fixture.mode = mode
                self.fixture.calls.clear()
                args = self.args("--execute")
                args.output_dir = Path(directory)
                run = cli.Backfill(self.client, args, clock=self.clock.monotonic, sleep=self.clock.sleep)
                run.run()
                row = run.manifest["documents"][0]
                self.assertFalse(row["complete"])
                self.assertEqual(row["action"], "needs_attention")
                self.assertFalse(any(call[1].endswith("/resume") for call in self.fixture.calls))
                self.assertNotIn("SECRET", run.path.read_text())

    def test_active_timeout_retains_run_id_for_resume(self):
        self.fixture.mode = "active"
        run = self.run_backfill("--execute")
        row = run.manifest["documents"][0]
        self.assertEqual(row["action"], "time_budget_reached_resume_later")
        self.assertEqual(row["run_id"], "run1")
        self.assertFalse(row["complete"])

    def test_no_progress_stops_instead_of_repeated_resume(self):
        self.fixture.mode = "stalled"
        # The stalled fixture must remain stalled after a queued resume.
        original = self.fixture.description
        def description(identifier, queued=False):
            if not queued:
                self.fixture.runs[identifier]["phase"] = "initial"
            return original(identifier, queued)
        self.fixture.description = description
        run = self.run_backfill("--execute")
        self.assertEqual(run.manifest["documents"][0]["action"], "needs_attention_no_progress")
        self.assertEqual(len([call for call in self.fixture.calls if call[1].endswith("/resume")]), 1)

    def test_old_or_unbound_provider_fails_before_any_mutation(self):
        self.fixture.provider["method"] = "old_provider"
        with self.assertRaisesRegex(cli.BackfillError, "improved_provider"):
            self.run_backfill("--execute")
        self.assertTrue(all(call[0] == "GET" for call in self.fixture.calls))
        self.fixture.provider["method"] = "staged_document_claims"
        self.fixture.ontology["extraction_guidance"]["execution_identity"]["model"] = "different-model"
        with self.assertRaisesRegex(cli.BackfillError, "not_bound"):
            self.run_backfill("--execute")

    def test_http_error_payload_never_escapes(self):
        class FailingHTTP:
            def open(self, request, timeout):
                raise HTTPError(request.full_url, 422, "private", {}, io.BytesIO(b'{"detail":"SECRET source text"}'))
        self.client.opener = FailingHTTP()
        with self.assertRaisesRegex(cli.BackfillError, "^http_422$"):
            self.client.request("/research-runs/from-extraction", {})

    def test_unsafe_base_and_duplicate_manifest_require_explicit_resume(self):
        with self.assertRaises(cli.BackfillError):
            cli.Client("https://user:SECRET@example.test/api/v1")
        self.run_backfill("--dry-run")
        with self.assertRaisesRegex(cli.BackfillError, "use_resume"):
            self.run_backfill("--execute")

    def test_supplied_papers_are_prioritized_before_larger_library(self):
        self.fixture.files = [
            {**self.fixture.files[0], "file_id": "book", "original_filename": "Book.pdf", "character_count": 900000},
            {**self.fixture.files[0], "file_id": "paper", "original_filename": "台湾地区树皮布石拍初探_邓聪.pdf", "character_count": 20000},
            {**self.fixture.files[0], "file_id": "short", "original_filename": "Small.pdf", "character_count": 1000},
        ]
        run = cli.Backfill(self.client, self.args("--dry-run"))
        tasks = run.plan()
        self.assertEqual([row["file_id"] for row, _ in tasks], ["paper", "short", "book"])


if __name__ == "__main__":
    unittest.main()
