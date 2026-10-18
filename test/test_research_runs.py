"""Hosted MVP graph API checks use temporary repository storage, no live model/network."""

import asyncio
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import repository, research_runs
from app.repository.knowledge_pipeline.jobs import GraphScheduler, SHARED_WORKSPACE, execution_lock
from app.repository.knowledge_pipeline.service import ResearchPipeline, validate_config
from app.repository.knowledge_pipeline.store import RunBusyError, RunStore, read_json, write_json
from test.test_knowledge_pipeline_service import ONTOLOGY, response_for


class ResearchRunApiTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for symbol, value in {"STORAGE_ROOT": self.root, "FILES_ROOT": self.root / "files",
                              "IMAGES_ROOT": self.root / "images", "DB_PATH": self.root / "repository.sqlite"}.items():
            self.stack.enter_context(patch.object(repository, symbol, str(value)))
        repository.init_repository_db()
        self.scheduler = GraphScheduler(RunStore(self.root / "research_runs", SHARED_WORKSPACE),
                                        pipeline_factory=lambda store: ResearchPipeline(store))
        self.stack.enter_context(patch.object(research_runs, "get_scheduler", return_value=self.scheduler))
        app = FastAPI()
        app.include_router(repository.router, prefix="/api/v1")
        app.include_router(research_runs.router, prefix="/api/v1")
        self.client = self.stack.enter_context(TestClient(app))

    def create(self, *, key="first", title="Barkcloth tools"):
        material = self.client.post("/api/v1/repository/materials", json={"title": "Synthetic source"}).json()["id"]
        upload = self.client.post(f"/api/v1/repository/materials/{material}/files", params={"filename": "paper.txt"},
                                  content=b"A stone beater processes barkcloth.")
        self.assertEqual(upload.status_code, 200, upload.text)
        body = {"title": title, "documents": [{"material_id": material, "file_id": upload.json()["id"]}],
                "config": {"ontology": ONTOLOGY, "augmentation": {"enabled": False}}}
        created = self.client.post("/api/v1/research-runs", json=body, headers={"Idempotency-Key": key})
        self.assertEqual(created.status_code, 201, created.text)
        return body, created.json()

    def process(self, run_id):
        self.assertTrue(asyncio.run(self.scheduler.process(run_id)))
        return self.scheduler.store.describe(run_id)

    def test_document_manual_concepts_review_release_and_restart_persistence(self):
        body, created = self.create()
        run_id = created["run_id"]
        duplicate = self.client.post("/api/v1/research-runs", json=body, headers={"Idempotency-Key": "first"})
        self.assertEqual(duplicate.json()["run_id"], run_id)
        self.assertEqual(duplicate.json()["job_id"], created["job_id"])
        self.assertEqual(self.process(run_id)["status"], "awaiting_annotations")
        endpoint = f"/api/v1/research-runs/{run_id}"
        batches = self.client.get(endpoint + "/artifacts/primary_batches.json").json()
        annotated = self.client.post(endpoint + "/annotations", json={"responses": {
            batches[0]["batch_id"]: response_for(batches[0]["units"])}})
        self.assertEqual(annotated.status_code, 200, annotated.text)
        self.assertEqual(self.client.post(endpoint + "/resume").status_code, 202)
        self.assertEqual(self.process(run_id)["status"], "awaiting_review")
        graph = self.client.get(endpoint + "/artifacts/graph_input.json").json()
        self.assertEqual(len(graph["entities"]), 2)
        self.assertEqual(len(graph["assertions"]), 1)
        self.assertEqual(graph["sources"][0]["source_locator"],
                         f"workbench:material/{body['documents'][0]['material_id']}/file/{body['documents'][0]['file_id']}")
        preview = self.client.post(endpoint + "/release", json={"allow_pending": True})
        self.assertEqual(preview.status_code, 201, preview.text)
        decisions = [{"candidate_id": row[key], "decision": "accept", "rationale": "Checked synthetic source quotation"}
                     for field, key in (("entities", "entity_id"), ("assertions", "assertion_id")) for row in graph[field]]
        self.assertEqual(self.client.post(endpoint + "/review", json={"decisions": decisions}).status_code, 200)
        released = self.client.post(endpoint + "/release", json={})
        self.assertEqual(released.status_code, 201, released.text)
        self.assertEqual(released.json()["status"], "released")
        release_path = released.json()["release_path"]
        self.assertEqual(self.client.get(endpoint + "/artifacts/" + release_path + "/graph.json").status_code, 200)
        self.assertEqual(self.client.get(endpoint + "/artifacts/" + release_path + "/graph.ttl").status_code, 200)
        reloaded = RunStore(self.root / "research_runs", SHARED_WORKSPACE)
        self.assertEqual(reloaded.describe(run_id)["status"], "released")
        self.assertEqual(reloaded.list_runs()["items"][0]["title"], "Barkcloth tools")
        self.assertEqual(self.client.post(endpoint + "/resume").status_code, 422)
        self.assertEqual(self.client.get(endpoint + "/artifacts/.job.json").status_code, 404)

    def test_explicit_current_processing_retry_preserves_original_and_is_idempotent(self):
        _, original = self.create()
        self.process(original["run_id"])
        original_path = self.scheduler.store.path(original["run_id"])
        prior = {name: (original_path / name).read_bytes() for name in ("inputs.json", "state.json", "primary_batches.json")}
        # An old implementation is deliberately not resumed/overwritten in place.
        old_inputs = read_json(original_path / "inputs.json")
        old_inputs["implementation_sha256"]["processing.py"] = "old-version"
        write_json(original_path / "inputs.json", old_inputs)
        prior["inputs.json"] = (original_path / "inputs.json").read_bytes()
        endpoint = f"/api/v1/research-runs/{original['run_id']}/retry"
        fresh = self.client.post(endpoint, json={})
        self.assertEqual(fresh.status_code, 201, fresh.text)
        self.assertNotEqual(fresh.json()["run_id"], original["run_id"])
        self.assertEqual(fresh.json()["parent_run_id"], original["run_id"])
        duplicate = self.client.post(endpoint, json={})
        self.assertEqual(duplicate.json()["run_id"], fresh.json()["run_id"])
        for name, data in prior.items():
            self.assertEqual((original_path / name).read_bytes(), data)
        self.assertEqual(self.client.post(endpoint, json={"config": {"processing": {"mode": "unsafe"}}}).status_code, 422)

    def test_versioned_ontology_templates_and_best_effort_capabilities(self):
        capability = self.client.get("/api/v1/research-runs/capabilities").json()
        self.assertTrue(capability["extraction"]["processing_version"])
        self.assertEqual({row["id"] for row in capability["ontology_templates"]}, {"general", "tapa"})
        general = self.client.get("/api/v1/research-runs/ontologies/general")
        self.assertEqual(general.status_code, 200, general.text)
        self.assertIn("Person", general.json()["entity_types"])
        self.assertIn("related_to", general.json()["predicates"])
        self.assertNotIn("ProductionCase", general.json()["entity_types"])
        self.assertEqual(self.client.get("/api/v1/research-runs/ontologies/tapa").status_code, 200)
        self.assertEqual(self.client.get("/api/v1/research-runs/ontologies/unknown").status_code, 422)
        validate_config({"ontology": general.json()})

    def test_discovery_request_is_a_queued_audited_action(self):
        _, created = self.create()
        run_id = created["run_id"]
        self.process(run_id)
        path = self.scheduler.store.path(run_id)
        self.assertEqual(self.client.post(f"/api/v1/research-runs/{run_id}/discover").status_code, 422)
        # Synthetic state transition; network discovery is tested with a mocked registry.
        self.scheduler.store.state(path, "awaiting_frontier", "augmentation")
        response = self.client.post(f"/api/v1/research-runs/{run_id}/discover")
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["job_status"], "queued")
        self.assertEqual(read_json(path / "discovery_request.json")["method"], "crossref")

    def test_graph_ingest_uses_registered_filename_and_language_for_shared_ocr(self):
        material = self.client.post("/api/v1/repository/materials", json={"title": "English catalogue title", "language": "ja"}).json()["id"]
        upload = self.client.post(f"/api/v1/repository/materials/{material}/files", params={"filename": "日本語.pdf"}, content=b"synthetic PDF bytes")
        run = self.client.post("/api/v1/research-runs", json={"documents": [{"material_id": material, "file_id": upload.json()["id"]}],
            "config": {"augmentation": {"enabled": False}}}).json()
        shared = {"representation_id": "japanese-page-v1", "method_version": "transcription-v1", "status": "text_ready", "warnings": [],
                  "coverage": {"unreadable_locators": []}, "units": [{"page_index": 1, "locator": "page:1", "text": "樹皮布の道具。", "method": "tesseract_ocr", "language": "jpn", "confidence": .9}]}
        with patch("app.repository.document_representation.extract_document", return_value=shared) as extract:
            self.assertEqual(self.process(run["run_id"])["status"], "awaiting_annotations")
        self.assertEqual(extract.call_args.args[1], "日本語.pdf")
        self.assertEqual(extract.call_args.kwargs["languages"], "ja")
        inputs = read_json(self.scheduler.store.path(run["run_id"]) / "inputs.json")
        self.assertEqual(inputs["documents"][0]["original_filename"], "日本語.pdf")
        self.assertEqual(inputs["documents"][0]["ocr_languages"], "ja")

    def test_scheduler_restart_does_not_reuse_an_old_event_loop(self):
        async def lifecycle():
            self.scheduler.start()
            await asyncio.sleep(.01)
            self.assertFalse(self.scheduler._task.done())
            await self.scheduler.stop()
        asyncio.run(lifecycle())
        asyncio.run(lifecycle())

    def test_input_validation_listing_and_protected_artifact_paths(self):
        _, first = self.create(key="one")
        _, second = self.create(key="two", title="Second graph")
        listing = self.client.get("/api/v1/research-runs", params={"limit": 1}).json()
        self.assertEqual(listing["total"], 2)
        self.assertEqual(listing["items"][0]["run_id"], second["run_id"])
        self.assertEqual(self.client.get("/api/v1/research-runs", params={"limit": 101}).status_code, 422)
        self.assertEqual(self.client.get("/api/v1/research-runs/not-a-uuid").status_code, 422)
        invalid = self.client.post("/api/v1/research-runs", json={"documents": [{"material_id": "missing", "file_id": "missing"}]})
        self.assertEqual(invalid.status_code, 404)
        escaped = self.client.get(f"/api/v1/research-runs/{first['run_id']}/artifacts/%2E%2E/repository.sqlite")
        self.assertEqual(escaped.status_code, 404)

    def test_material_filter_matches_exact_ids_before_pagination(self):
        source = self.root / "document.txt"
        source.write_text("A stone beater processes barkcloth.")
        config = validate_config({"ontology": ONTOLOGY, "augmentation": {"enabled": False}})
        def create_for(material_ids, title):
            return self.scheduler.store.create([
                {"path": source, "locator": f"workbench:material/{material_id}/file/file-1"}
                for material_id in material_ids
            ], config, "fixture", title=title)
        first = create_for(["doc-1"], "First matching graph")
        second = create_for(["other", "doc-1"], "Second matching graph")
        create_for(["doc-10"], "Prefix must not match")
        create_for(["unrelated"], "Newer unrelated graph")
        endpoint = "/api/v1/research-runs"
        page = self.client.get(endpoint, params={"material_id": "doc-1", "limit": 1}).json()
        self.assertEqual(page["total"], 2)
        self.assertEqual([item["run_id"] for item in page["items"]], [second["run_id"]])
        page = self.client.get(endpoint, params={"material_id": "doc-1", "limit": 1, "offset": 1}).json()
        self.assertEqual(page["total"], 2)
        self.assertEqual([item["run_id"] for item in page["items"]], [first["run_id"]])
        self.assertEqual(self.client.get(endpoint).json()["total"], 4)
        self.assertEqual(self.client.get(endpoint, params={"material_id": "doc"}).json()["total"], 0)
        for invalid in ("", "x" * 256, "doc-1/file"):
            self.assertEqual(self.client.get(endpoint, params={"material_id": invalid}).status_code, 422)
            with self.assertRaises(ValueError):
                self.scheduler.store.list_runs(material_id=invalid)

    def test_capabilities_reuse_hosted_model_configuration_without_secrets(self):
        with patch.object(repository, "ai_chat_configured", return_value=True), patch.object(research_runs, "find_spec", return_value=object()):
            capabilities = self.client.get("/api/v1/research-runs/capabilities")
        self.assertEqual(capabilities.status_code, 200)
        self.assertEqual(capabilities.json()["extraction"]["default_mode"], "ollama")
        self.assertTrue(capabilities.json()["collection"]["available"])
        self.assertNotIn("Authorization", capabilities.text)
        self.assertNotIn("base_url", capabilities.text)
        with patch.object(repository, "ai_chat_configured", return_value=False), patch.object(research_runs, "find_spec", return_value=None), patch("app.repository.knowledge_pipeline.runtime.APPROVED_LOCAL_RUNTIME", self.root / "missing"):
            capabilities = self.client.get("/api/v1/research-runs/capabilities").json()
        self.assertEqual(capabilities["extraction"]["default_mode"], "manual")
        self.assertFalse(capabilities["collection"]["available"])

    def test_persisted_queued_and_interrupted_jobs_recover_without_progress_tables(self):
        _, created = self.create()
        run_id = created["run_id"]
        path = self.scheduler.store.path(run_id)
        interrupted = read_json(path / ".job.json")
        interrupted["status"] = "running"
        write_json(path / ".job.json", interrupted)
        restarted = GraphScheduler(RunStore(self.root / "research_runs", SHARED_WORKSPACE),
                                   pipeline_factory=lambda store: ResearchPipeline(store))
        self.assertIn(run_id, restarted.pending())
        self.assertTrue(asyncio.run(restarted.process(run_id)))
        self.assertEqual(restarted.store.describe(run_id)["status"], "awaiting_annotations")
        self.assertEqual(read_json(path / ".job.json")["status"], "completed")
        self.assertEqual(read_json(path / ".job.json")["attempts"], 1)
        self.assertFalse(asyncio.run(restarted.process(run_id)))
        self.assertEqual(restarted.pending(), [])

    def test_duplicate_queue_and_execution_lock_prevent_duplicate_work(self):
        _, created = self.create()
        run_id = created["run_id"]
        self.assertEqual(self.scheduler.enqueue(run_id)["job_id"], created["job_id"])
        with execution_lock(self.scheduler.store.path(run_id)) as acquired:
            self.assertTrue(acquired)
            self.assertFalse(asyncio.run(self.scheduler.process(run_id)))
        self.assertTrue(asyncio.run(self.scheduler.process(run_id)))

    def test_pre_pipeline_failure_updates_visible_run_state_and_can_resume(self):
        _, created = self.create()
        run_id = created["run_id"]
        def broken_factory(store):
            raise RuntimeError("PRIVATE_CONFIG_DETAIL")
        self.scheduler.pipeline_factory = broken_factory
        self.assertTrue(asyncio.run(self.scheduler.process(run_id)))
        endpoint = f"/api/v1/research-runs/{run_id}"
        failed = self.client.get(endpoint)
        self.assertEqual(failed.json()["status"], "failed")
        self.assertNotIn("PRIVATE_CONFIG_DETAIL", failed.text)
        self.assertEqual(self.scheduler.pending(), [])
        self.scheduler.pipeline_factory = lambda store: ResearchPipeline(store)
        self.assertEqual(self.client.post(endpoint + "/resume").status_code, 202)
        self.assertEqual(self.process(run_id)["status"], "awaiting_annotations")

    def test_transient_run_lock_contention_remains_queued(self):
        _, created = self.create()
        run_id = created["run_id"]
        class ContendedPipeline:
            async def run(self, ignored):
                raise RunBusyError("Another request is updating this run")
        self.scheduler.pipeline_factory = lambda store: ContendedPipeline()
        self.assertFalse(asyncio.run(self.scheduler.process(run_id)))
        self.assertEqual(self.scheduler.pending(), [run_id])
        self.assertEqual(self.scheduler.store.describe(run_id)["status"], "queued")
        self.scheduler.pipeline_factory = lambda store: ResearchPipeline(store)
        self.assertEqual(self.process(run_id)["status"], "awaiting_annotations")


class SchedulerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_processes_persisted_job_and_shutdown_is_clean(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            doc = root / "doc.txt"
            doc.write_text("A stone beater processes barkcloth.")
            store = RunStore(root / "runs", SHARED_WORKSPACE)
            run = store.create([{"path": doc}], validate_config({"ontology": ONTOLOGY, "augmentation": {"enabled": False}}), "fixture")
            original = GraphScheduler(store)
            original.enqueue(run["run_id"])
            scheduler = GraphScheduler(store, pipeline_factory=lambda target: ResearchPipeline(target), poll_seconds=0.01)
            scheduler.start()
            try:
                for _ in range(100):
                    if store.describe(run["run_id"])["status"] == "awaiting_annotations":
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(store.describe(run["run_id"])["status"], "awaiting_annotations")
            finally:
                await scheduler.stop()
            self.assertIsNone(scheduler._task)

    async def test_cancelled_processing_is_recovered_by_a_new_scheduler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            doc = root / "doc.txt"
            doc.write_text("A stone beater processes barkcloth.")
            store = RunStore(root / "runs", SHARED_WORKSPACE)
            run = store.create([{"path": doc}], validate_config({"ontology": ONTOLOGY, "augmentation": {"enabled": False}}), "fixture")
            entered = asyncio.Event()
            class WaitingPipeline:
                async def run(self, run_id):
                    entered.set()
                    await asyncio.Event().wait()
            scheduler = GraphScheduler(store, pipeline_factory=lambda target: WaitingPipeline())
            scheduler.enqueue(run["run_id"])
            scheduler.start()
            await asyncio.wait_for(entered.wait(), 1)
            await scheduler.stop()
            self.assertEqual(store.describe(run["run_id"])["status"], "queued")
            recovered = GraphScheduler(store, pipeline_factory=lambda target: ResearchPipeline(target))
            self.assertIn(run["run_id"], recovered.pending())
            self.assertTrue(await recovered.process(run["run_id"]))
            self.assertEqual(store.describe(run["run_id"])["status"], "awaiting_annotations")
