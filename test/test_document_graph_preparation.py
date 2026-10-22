"""Automatic graph preparation uses exact current snapshots without source caps."""
import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch

from app.api import repository
from app.repository.knowledge_pipeline.jobs import GraphScheduler, SHARED_WORKSPACE
from app.repository.knowledge_pipeline.service import ResearchPipeline
from app.repository.knowledge_pipeline.store import RunStore, read_json
from app.repository.processing_jobs import ProcessingScheduler
from test import test_document_graph_inventory as fixtures


class DocumentGraphPreparationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DocumentGraphInventoryTests("test_unknown_material_has_no_eligible_input")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.graphs = GraphScheduler(RunStore(self.root / "research_runs", SHARED_WORKSPACE),
                                     pipeline_factory=lambda store: ResearchPipeline(store))
        self.preparation = ProcessingScheduler(self.root / "processing_jobs")
        self.job = {"material_id": self.fixture.material, "input_hash": "unchanged-original-bytes-and-method"}
        self.provider_patch = patch.object(repository, "ai_chat_configured", return_value=False)
        self.provider_patch.start()
        self.addCleanup(self.provider_patch.stop)
        self.scheduler_patch = patch("app.repository.document_graph_runtime.get_scheduler", return_value=self.graphs)
        self.scheduler_patch.start()
        self.addCleanup(self.scheduler_patch.stop)

    def prepare(self):
        return asyncio.run(self.preparation.run_stage("graph", self.job))

    def test_identical_snapshot_reuses_an_already_started_run_without_requeuing(self):
        first = self.prepare()
        run_id = first["run_id"]
        self.assertTrue(asyncio.run(self.graphs.process(run_id)))
        before = self.graphs.store.describe(run_id)
        self.assertEqual(before["status"], "awaiting_annotations")
        path = self.graphs.store.path(run_id)
        job_before = (path / ".job.json").read_bytes()
        repeated = self.prepare()
        self.assertEqual(repeated["run_id"], run_id)
        self.assertEqual(repeated["status"], "awaiting_annotations")
        self.assertEqual((path / ".job.json").read_bytes(), job_before)
        self.assertEqual(self.graphs.store.list_runs()["total"], 1)

    def test_new_extraction_metadata_creates_a_new_run_for_unchanged_original_bytes(self):
        original = self.root / "files" / self.fixture.file
        original_bytes = original.read_bytes()
        first = self.prepare()
        first_path = self.graphs.store.path(first["run_id"])
        before_inputs = (first_path / "inputs.json").read_bytes()
        first_inputs = read_json(first_path / "inputs.json")
        with repository.get_connection() as connection:
            connection.execute("INSERT INTO extraction_runs (id,material_id,include_links,max_link_depth,max_link_pages,extracted_segment_count,discovered_link_count,status,error_message,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                               ("extraction-new", self.fixture.material, 0, 0, 0, 2, 0, "ok", None,
                                "2026-10-22T02:00:00Z"))
        second = self.prepare()
        self.assertNotEqual(second["run_id"], first["run_id"])
        second_path = self.graphs.store.path(second["run_id"])
        second_inputs = read_json(second_path / "inputs.json")
        self.assertNotEqual(first_inputs["documents"][0]["sha256"], second_inputs["documents"][0]["sha256"])
        self.assertEqual((first_path / "inputs.json").read_bytes(), before_inputs)
        self.assertEqual(original.read_bytes(), original_bytes)
        first_representation = read_json(first_path / first_inputs["documents"][0]["input_path"])
        second_representation = read_json(second_path / second_inputs["documents"][0]["input_path"])
        self.assertEqual(first_representation["segments"], second_representation["segments"])
        self.assertEqual(first_representation["parent_file"], second_representation["parent_file"])
        self.assertEqual(second_representation["extraction_run"]["id"], "extraction-new")
        self.assertEqual(self.graphs.store.list_runs()["total"], 2)

    def test_all_registered_source_pages_are_retained_without_a_two_page_cap(self):
        for page in range(3, 36):
            self.fixture.add_segment(self.fixture.file, f"Original source wording on page {page}.", page=page)
        original_files = {path.name: path.read_bytes() for path in (self.root / "files").iterdir() if path.is_file()}
        created = self.prepare()
        path = self.graphs.store.path(created["run_id"])
        inputs = read_json(path / "inputs.json")
        self.assertEqual(inputs["documents"][0]["representation_kind"], "registered_extraction")
        self.assertIsNone(inputs["config"]["processing"]["max_auto_batches"])
        payload = read_json(path / inputs["documents"][0]["input_path"])
        self.assertEqual(len(payload["segments"]), 35)
        self.assertEqual({row["page_index"] for row in payload["segments"]}, set(range(1, 36)))
        self.assertTrue(asyncio.run(self.graphs.process(created["run_id"])))
        ingest = read_json(path / "ingest.json")
        self.assertEqual(len(ingest["units"]), 35)
        self.assertEqual({row["locator"]["page_index"] for row in ingest["units"]}, set(range(1, 36)))
        self.assertEqual([row["text"] for row in ingest["units"]], [row["content_text"] for row in payload["segments"]])
        self.assertEqual({path.name: path.read_bytes() for path in (self.root / "files").iterdir() if path.is_file()}, original_files)
        self.assertEqual(ingest["sources"][0]["source_locator"],
                         f"workbench:material/{self.fixture.material}/file/{self.fixture.file}")
