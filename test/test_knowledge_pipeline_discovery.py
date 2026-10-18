"""Synthetic URL discovery, rolling stop, replay and graph-search isolation checks."""
import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from app.repository.knowledge_pipeline.adapters import ingest_document
from app.repository.knowledge_pipeline.discovery import discover_frontier, public_url
from app.repository.knowledge_pipeline.processing import process_response
from app.repository.knowledge_pipeline.search import search_graph_entities
from app.repository.knowledge_pipeline.service import ResearchPipeline, validate_config
from app.repository.knowledge_pipeline.store import RunStore, read_json, write_json
from test.test_knowledge_pipeline_service import ONTOLOGY, response_for


class DiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_saves_bounded_registry_response_and_exact_seed_lineage(self):
        seen = []
        def transport(request):
            seen.append(request)
            return httpx.Response(200, json={"message": {"items": [
                {"title": ["Stone beater barkcloth study"], "URL": "https://example.org/beater", "DOI": "10.1000/fixture"},
                {"title": ["Stone beater on private network"], "URL": "http://127.0.0.1/private"},
                {"title": ["Unrelated protein study"], "URL": "https://example.org/protein"},
            ]}})
        client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        with tempfile.TemporaryDirectory() as temp, patch("app.repository.knowledge_pipeline.discovery.httpx.AsyncClient", return_value=client):
            output = Path(temp) / "discovery.json"
            seeds = [{"seed_id": f"seed{i}", "label": "stone beater", "question": "How is the tool used?"} for i in range(8)]
            result = await discover_frontier(seeds, "barkcloth", output, 2)
            self.assertEqual(len(seen), 5)
            self.assertEqual(len(result["frontier"]), 1)
            self.assertEqual(result["frontier"][0]["seed_ids"], [f"seed{i}" for i in range(5)])
            self.assertEqual(result["frontier"][0]["url"], "https://example.org/beater")
            self.assertEqual(read_json(output), result)
            expected_bytes = json.dumps(result["attempts"][0]["response"], ensure_ascii=False, separators=(",", ":")).encode()
            self.assertEqual(result["attempts"][0]["response_sha256"], hashlib.sha256(expected_bytes).hexdigest())
            self.assertTrue(all(row["response_sha256"] for row in result["attempts"]))

    async def test_discovery_errors_are_saved_without_inventing_frontier(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(429, json={"error": "not source evidence"})))
        with tempfile.TemporaryDirectory() as temp, patch("app.repository.knowledge_pipeline.discovery.httpx.AsyncClient", return_value=client):
            result = await discover_frontier([{"seed_id": "seed", "label": "stone beater", "question": "how?"}], "barkcloth", Path(temp) / "discovery.json", 1)
        self.assertEqual(result["frontier"], [])
        self.assertEqual(result["attempts"][0]["http_status"], 429)
        for url in ("http://localhost/x", "http://127.0.0.1/x", "file:///x", "http://[bad", "https://u:p@example.org/x"):
            self.assertIsNone(public_url(url))


class RollingAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.document = self.root / "source.txt"
        self.document.write_text("A stone beater processes barkcloth.")
        self.store = RunStore(self.root / "runs", "fixture")
        self.calls = {"provider": 0, "collector": 0}
        async def provider(units, ontology):
            self.calls["provider"] += 1
            response = response_for(units)
            return process_response(response, units, ontology)
        def collector(frontier, destination, options):
            self.calls["collector"] += 1
            sources, units, attempts = [], [], []
            for i, row in enumerate(frontier):
                source_id = "WEB_" + row["url"].rsplit("/", 1)[-1]
                item = ingest_document(self.document, source_id, "Synthetic independent source", destination / str(i), origin="web")
                item["source"].update(seed_ids=row["seed_ids"], source_locator=row["url"])
                for unit in item["units"]:
                    unit["seed_ids"] = row["seed_ids"]
                sources.append(item["source"])
                units.extend(item["units"])
                attempts.append({"url": row["url"], "status": "collected"})
            return {"sources": sources, "units": units, "attempts": attempts}
        self.pipeline = ResearchPipeline(self.store, provider, collector)

    def test_low_yield_windows_stop_before_page_budget_and_replay_reuses_snapshots(self):
        config = validate_config({"ontology": ONTOLOGY, "processing": {"mode": "ollama"}, "augmentation": {
            "max_pages": 6, "window_size": 2, "min_successful_pages": 4, "low_yield_windows": 2}})
        run = self.store.create([{"path": self.document}], config, "fixture")
        self.assertEqual(asyncio.run(self.pipeline.run(run["run_id"]))["status"], "awaiting_frontier")
        path = self.store.path(run["run_id"])
        seeds = [row["seed_id"] for row in read_json(path / "seeds.json")]
        self.pipeline.submit_frontier(run["run_id"], [{"url": f"https://example.org/{i}", "seed_ids": seeds} for i in range(6)], "fixture")
        state = asyncio.run(self.pipeline.run(run["run_id"]))
        self.assertEqual(state["status"], "awaiting_review")
        stop = read_json(path / "augmentation_metrics.json")["stopping"]
        self.assertEqual(stop["reason"], "rolling_low_yield")
        self.assertEqual(stop["successful_pages"], 4)
        self.assertEqual(stop["remaining_pages"], 2)
        self.assertEqual(self.calls["collector"], 2)
        calls = dict(self.calls)
        asyncio.run(self.pipeline.run(run["run_id"]))
        self.assertEqual(self.calls, calls)
        relevance = read_json(path / "augmentation_relevance.json")["entities"]
        self.assertEqual(len(relevance), 8)
        self.assertTrue(all(item["relevance_status"] == "needs_specialist_review" for item in relevance))
        self.assertTrue(all(not item["identity_or_fact_transfer"] for item in relevance))

    def test_graph_search_material_filter_does_not_leak_other_document_entities(self):
        config = validate_config({"ontology": ONTOLOGY, "processing": {"mode": "ollama"}, "augmentation": {"enabled": False}})
        run = self.store.create([{"path": self.document, "locator": "workbench:material/doc-a/file/a"},
                                 {"path": self.document, "locator": "workbench:material/doc-b/file/b"}], config, "fixture")
        asyncio.run(self.pipeline.run(run["run_id"]))
        all_hits = search_graph_entities("stone beater", store=self.store)
        a_hits = search_graph_entities("stone beater", material_id="doc-a", store=self.store)
        self.assertEqual(len(all_hits), 2)
        self.assertEqual(len(a_hits), 1)
        self.assertEqual(a_hits[0]["source_locator"], "workbench:material/doc-a/file/a")
        self.assertEqual(a_hits[0]["publication_status"], "draft")
        self.assertEqual(a_hits[0]["scientific_status"], "needs_specialist_review")
        self.assertTrue(a_hits[0]["evidence"])

    def test_config_rejects_unbounded_or_malformed_discovery_and_stop_options(self):
        for option in ({"discovery": []}, {"window_size": 0}, {"min_successful_pages": 26},
                       {"low_yield_windows": True}, {"min_avg_novel_entities": float("nan")}, {"duplicate_ceiling": 2}):
            with self.assertRaises(ValueError):
                validate_config({"augmentation": option})
