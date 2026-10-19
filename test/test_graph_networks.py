"""No model/network calls: graph combination, provenance and database coverage."""
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import copy
import errno
import hashlib
from pathlib import Path
import tempfile
import types
from threading import Barrier
import unittest
from unittest.mock import patch
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import graph_networks, repository
from app.repository.graph_networks import NetworkStore, _all_runs, load_saved_graph, namespace_graph
from app.repository import graph_networks as network_storage
from app.repository.knowledge_pipeline.jobs import SHARED_WORKSPACE
from app.repository.knowledge_pipeline.store import RunStore, read_json, write_json


class GraphNetworkTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for symbol, value in {"STORAGE_ROOT": self.root, "FILES_ROOT": self.root / "files",
                              "IMAGES_ROOT": self.root / "images", "DB_PATH": self.root / "repository.sqlite"}.items():
            self.stack.enter_context(patch.object(repository, symbol, str(value)))
        repository.init_repository_db()
        self.runs = RunStore(self.root / "research_runs", SHARED_WORKSPACE)
        self.networks = NetworkStore(self.root / "graph_networks", self.runs)
        self.stack.enter_context(patch.object(graph_networks, "shared_network_store", return_value=self.networks))
        app = FastAPI()
        app.include_router(repository.router, prefix="/api/v1")
        app.include_router(graph_networks.router, prefix="/api/v1")
        self.client = self.stack.enter_context(TestClient(app))

    def source(self, title="Barkcloth paper"):
        material = self.client.post("/api/v1/repository/materials", json={"title": title}).json()["id"]
        file = self.client.post(f"/api/v1/repository/materials/{material}/files", params={"filename": "paper.txt"},
                                content="A stone beater processes barkcloth.".encode()).json()
        with repository.get_connection() as connection:
            row = connection.execute("SELECT stored_path FROM files WHERE id=?", (file["id"],)).fetchone()
        return material, file["id"], Path(row["stored_path"])

    def create_run(self, title="Barkcloth paper", *, partial=False):
        material, file, original = self.source(title)
        locator = f"workbench:material/{material}/file/{file}"
        created = self.runs.create([{"path": original, "title": title, "locator": locator}], {"ontology": {"version": "test"}}, "test")
        path = self.runs.path(created["run_id"])
        graph = {
            "sources": [{"source_id": "DOC_001", "source_version_id": "DOC_001:version", "source_locator": locator, "title": title,
                         "seed_ids": ["E1"], "seeds": [{"seed_id": "SEED1", "entity_id": "E1"}]}],
            "entities": [{"entity_id": "E1", "entity_type": "Concept", "label_original": "stone beater", "source_id": "DOC_001",
                          "source_version_id": "DOC_001:version", "evidence_ids": ["EV1"], "scope": {"place_id": "E2"},
                          "review_status": "accepted", "candidate_version": "old_candidate"},
                         {"entity_id": "E2", "entity_type": "Concept", "label_original": "barkcloth", "source_id": "DOC_001", "evidence_ids": ["EV1"]}],
            "evidence": [{"evidence_id": "EV1", "source_id": "DOC_001", "source_version_id": "DOC_001:version", "unit_id": "DOC_001:unit:1",
                          "text": "stone beater", "text_sha256": hashlib.sha256(b"stone beater").hexdigest(),
                          "evidence_version_id": "EV1:version", "locator": {"kind": "pdf_page", "pdf_page": 1, "material_id": material, "file_id": file},
                          "character_locator": {"spans": [{"start": 2, "end": 14}], "offset_unit": "unicode_code_points"}}],
            "assertions": [{"assertion_id": "A1", "subject_id": "E1", "predicate": "related_to", "object_id": "E2", "source_id": "DOC_001",
                            "evidence_ids": ["EV1"], "evidence_version_ids": {"EV1": "EV1:version"}, "scope": {"case_id": "E2", "seed_ids": ["E1"]}, "polarity": "positive"},
                           {"assertion_id": "A2", "subject_id": "E1", "predicate": "description", "value": "E2", "source_id": "DOC_001", "evidence_ids": ["EV1"], "scope": {}, "polarity": "positive"}],
            "alignments": [], "partial_extraction": partial,
        }
        write_json(path / "graph_input.json", graph)
        self.runs.state(path, "awaiting_review", "release")
        return created["run_id"], material, file, original, graph

    def test_combination_namespaces_all_bound_ids_preserves_originals_and_is_persistent(self):
        first, material, file, _, graph = self.create_run()
        second, _, _, _, _ = self.create_run("Second paper")
        original = copy.deepcopy(graph)
        result = self.client.post("/api/v1/graph-networks", json={"title": "Comparison", "run_ids": [first, second], "infer_connections": False})
        self.assertEqual(result.status_code, 201, result.text)
        network = result.json()
        self.assertEqual(network["counts"]["entities"], 4)
        snapshot = self.client.get(f"/api/v1/graph-networks/{network['network_id']}/graph").json()
        self.assertEqual(len({row["source_id"] for row in snapshot["sources"]}), 2)
        prefix = f"RUN:{first}:"
        first_entity = next(row for row in snapshot["entities"] if row["entity_id"] == prefix + "E1")
        self.assertEqual(first_entity["scope"]["place_id"], prefix + "E2")
        self.assertEqual(first_entity["review_status"], "needs_specialist_review")
        self.assertEqual(first_entity["candidate_version"], prefix + "old_candidate")
        evidence = next(row for row in snapshot["evidence"] if row["source_id"] == prefix + "DOC_001")
        self.assertEqual(evidence["text"], original["evidence"][0]["text"])
        self.assertEqual(evidence["text_sha256"], original["evidence"][0]["text_sha256"])
        self.assertEqual(evidence["character_locator"], original["evidence"][0]["character_locator"])
        self.assertEqual(evidence["locator"]["material_id"], material)
        self.assertEqual(evidence["locator"]["file_id"], file)
        self.assertEqual(evidence["source_version_id"], prefix + "DOC_001:version")
        assertion = next(row for row in snapshot["assertions"] if row["assertion_id"] == prefix + "A1")
        self.assertEqual(assertion["evidence_version_ids"], {prefix + "EV1": prefix + "EV1:version"})
        self.assertEqual(assertion["scope"]["seed_ids"], [prefix + "E1"])
        literal = next(row for row in snapshot["assertions"] if row["assertion_id"] == prefix + "A2")
        self.assertEqual(literal["value"], "E2")
        self.assertEqual(snapshot["decisions"], [])
        self.assertEqual(read_json(self.runs.path(first) / "graph_input.json"), original)
        repeated = self.networks.create("Comparison", [second, first], infer_connections=False)
        self.assertEqual(repeated["network_id"], network["network_id"])
        persisted = NetworkStore(self.root / "graph_networks", self.runs).get(network["network_id"])
        self.assertEqual(persisted["graph"], snapshot)
        self.assertEqual(self.client.get("/api/v1/graph-networks").json()["total"], 1)
        exported = self.client.get(f"/api/v1/graph-networks/{network['network_id']}/export")
        self.assertIn("attachment", exported.headers["content-disposition"])
        self.assertEqual(exported.json()["network"]["network_id"], network["network_id"])

    def test_library_snapshot_includes_missing_documents_and_only_best_graph(self):
        first, material, _, _, graph = self.create_run(partial=True)
        _, missing, _ = self.source("Unprocessed paper")
        empty = self.client.post("/api/v1/repository/materials", json={"title": "Reference without file"}).json()["id"]
        snapshot = self.networks.create("Library database", [], "library", False)
        coverage = snapshot["coverage"]
        self.assertEqual(coverage["material_count"], 3)
        self.assertEqual(coverage["materials_with_graph"], 1)
        self.assertEqual(coverage["materials_without_graph"], 2)
        self.assertEqual(coverage["covered_file_count"], 1)
        self.assertEqual(snapshot["run_ids"], [first])
        self.assertTrue(next(row for row in coverage["materials"] if row["material_id"] == material)["partial_extraction"])
        self.assertEqual(next(row for row in coverage["materials"] if row["material_id"] == empty)["status"], "no_extracted_text")
        self.assertTrue(snapshot["partial_extraction"])

    def test_rejects_duplicate_absent_stale_or_altered_inputs(self):
        run_id, _, _, original, _ = self.create_run()
        for ids in ([run_id, run_id], [], [str(uuid.uuid4())], ["../bad"]):
            response = self.client.post("/api/v1/graph-networks", json={"title": "Invalid", "run_ids": ids, "infer_connections": False})
            self.assertIn(response.status_code, {404, 422}, response.text)
        original.write_text("Replaced source", encoding="utf-8")
        response = self.client.post("/api/v1/graph-networks", json={"title": "Stale", "run_ids": [run_id], "infer_connections": False})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("replaced", response.json()["detail"])

    def test_every_checkpoint_is_loaded_and_rejected_batches_are_not_accepted(self):
        run_id, _, _, _, graph = self.create_run()
        path = self.runs.path(run_id)
        (path / "graph_input.json").unlink()
        write_json(path / "ingest.json", {"sources": graph["sources"]})
        for index in range(23):
            write_json(path / "processed" / f"batch-{index:03d}.json", {"entities": [], "assertions": [], "evidence": []})
        write_json(path / "processed" / "batch-024.json", {field: graph[field] for field in ("entities", "assertions", "evidence")})
        write_json(path / "processed" / "rejected.json", {"entities": [{"entity_id": "bad"}], "rejections": [{"reason": "bad"}]})
        loaded, origin = load_saved_graph(self.runs, run_id)
        self.assertEqual(len(loaded["entities"]), 2)
        self.assertEqual(len(origin["artifacts"]), 24)
        self.assertEqual(loaded["sources"][0]["source_locator"], graph["sources"][0]["source_locator"])

    def test_upstream_rejection_is_preserved_as_provenance_but_approval_is_not_active(self):
        run_id, _, _, _, _ = self.create_run()
        write_json(self.runs.path(run_id) / "decisions.json", [
            {"candidate_id": "E1", "decision": "accept", "at": "2026-10-18T00:00:00Z"},
            {"candidate_id": "E1", "decision": "reject", "at": "2026-10-19T00:00:00Z"}])
        result = self.networks.create("Reviewed origin", [run_id], infer_connections=False)
        graph = self.networks.get(result["network_id"])["graph"]
        self.assertEqual(graph["decisions"], [])
        self.assertEqual(len(graph["origin_decisions"]), 1)
        self.assertEqual(graph["origin_decisions"][0]["decision"], "reject")
        self.assertEqual(graph["origin_decisions"][0]["candidate_id"], f"RUN:{run_id}:E1")

    def test_inferences_remain_separate_and_are_derived_after_namespacing(self):
        run_id, _, _, _, _ = self.create_run()
        seen = []
        def infer(graph):
            seen.extend(graph["entities"])
            return {"inferences": [{"inference_id": "INF:1", "subject_id": graph["entities"][0]["entity_id"],
                                     "object_id": graph["entities"][1]["entity_id"], "predicate": "context", "evidence_ids": [], "scope": {}}],
                    "report": {"added_connections": 1}}
        with patch.dict("sys.modules", {"app.repository.graph_inferences": types.SimpleNamespace(infer_connections=infer)}):
            result = self.networks.create("Inferred network", [run_id])
        graph = self.networks.get(result["network_id"])["graph"]
        self.assertTrue(seen[0]["entity_id"].startswith(f"RUN:{run_id}:"))
        self.assertEqual(result["counts"]["inferences"], 1)
        self.assertEqual(len(graph["assertions"]), 2)
        self.assertEqual(graph["alignments"], [])

    def test_all_pages_are_read(self):
        class Store:
            def list_runs(self, limit, offset):
                return {"items": list(range(offset, min(offset + limit, 203))), "total": 203}
        self.assertEqual(len(_all_runs(Store())), 203)

    def test_external_context_codes_are_preserved_and_original_review_uses_actual_instants(self):
        run_id, _, _, _, graph = self.create_run()
        graph["entities"][0]["scope"]["language_id"] = "fj"
        graph["entities"][0]["scope"]["external_catalog_id"] = "CAT:1"
        namespaced = namespace_graph(graph, run_id)
        self.assertEqual(namespaced["entities"][0]["scope"]["language_id"], "fj")
        self.assertEqual(namespaced["entities"][0]["scope"]["external_catalog_id"], "CAT:1")
        self.assertEqual(namespaced["entities"][0]["scope"]["place_id"], f"RUN:{run_id}:E2")
        write_json(self.runs.path(run_id) / "decisions.json", [
            {"candidate_id": "E1", "decision": "reject", "at": "2026-10-19T10:00:00+09:00"},
            {"candidate_id": "E1", "decision": "accept", "at": "2026-10-19T02:00:00Z"}])
        loaded, _ = load_saved_graph(self.runs, run_id)
        self.assertEqual(loaded["origin_decisions"][0]["decision"], "accept")

    def test_dangling_context_and_expired_source_cannot_contribute_a_new_network(self):
        run_id, _, _, _, graph = self.create_run()
        path = self.runs.path(run_id)
        graph["entities"][0]["scope"]["place_id"] = "unresolved-context"
        write_json(path / "graph_input.json", graph)
        with self.assertRaisesRegex(ValueError, "context reference"):
            self.networks.create("Invalid context", [run_id], infer_connections=False)
        graph["entities"][0]["scope"]["place_id"] = "E2"
        graph["sources"][0]["retention_expires_at"] = "2000-01-01T00:00:00Z"
        write_json(path / "graph_input.json", graph)
        with self.assertRaisesRegex(ValueError, "retention has expired"):
            self.networks.create("Expired source", [run_id], infer_connections=False)

    def test_attached_file_without_entities_is_reported_as_uncovered(self):
        run_id, material, _, _, _ = self.create_run()
        upload = self.client.post(f"/api/v1/repository/materials/{material}/files", params={"filename": "unprocessed.txt"}, content=b"An unprocessed second file.")
        self.assertEqual(upload.status_code, 200)
        snapshot = self.networks.create("Partial file coverage", [run_id], infer_connections=False)
        coverage = snapshot["coverage"]
        self.assertEqual(coverage["file_count"], 2)
        self.assertEqual(coverage["covered_file_count"], 1)
        self.assertTrue(snapshot["partial_extraction"])
        omitted = [row for row in coverage["materials"][0]["files"] if not row["included_in_graph"]]
        self.assertEqual([row["original_filename"] for row in omitted], ["unprocessed.txt"])

    def test_published_snapshot_is_authoritative_and_missing_release_never_uses_drafts(self):
        run_id, _, _, _, graph = self.create_run()
        path = self.runs.path(run_id)
        published = copy.deepcopy(graph)
        published["entities"] = [published["entities"][1]]
        published["assertions"] = []
        published["sources"][0]["title"] = "Authoritative source title"
        write_json(path / "releases" / "version1" / "graph.json", published)
        write_json(path / "processed" / "extra.json", {field: graph[field] for field in ("entities", "assertions", "evidence")})
        self.runs.state(path, "released", "release", release_path="releases/version1")
        loaded, origin = load_saved_graph(self.runs, run_id)
        for field in ("sources", "entities", "assertions", "alignments", "evidence"):
            self.assertEqual(loaded[field], published[field])
        self.assertEqual([row["name"] for row in origin["artifacts"]], ["releases/version1/graph.json"])
        (path / "releases" / "version1" / "graph.json").unlink()
        with self.assertRaisesRegex(ValueError, "draft checkpoints cannot replace"):
            load_saved_graph(self.runs, run_id)

    def test_capacity_failure_returns_sanitized_503_and_leaves_no_incomplete_snapshot(self):
        run_id, _, _, _, _ = self.create_run()
        with patch("app.repository.processing_jobs.shutil.disk_usage", return_value=types.SimpleNamespace(free=0)):
            response = self.client.post("/api/v1/graph-networks", json={"title": "Low storage", "run_ids": [run_id], "infer_connections": False})
        self.assertEqual(response.status_code, 503, response.text)
        self.assertNotIn(str(self.root), response.text)
        self.assertEqual(list(self.networks.root.glob("*.json")), [])
        self.assertEqual(list(self.networks.root.glob("*.tmp")), [])
        with patch.object(Path, "replace", side_effect=OSError(errno.ENOSPC, "private filesystem detail")):
            response = self.client.post("/api/v1/graph-networks", json={"title": "Write failure", "run_ids": [run_id], "infer_connections": False})
        self.assertEqual(response.status_code, 503, response.text)
        self.assertNotIn("private filesystem detail", response.text)
        self.assertEqual(list(self.networks.root.glob("*.tmp")), [])
        self.assertEqual(list(self.networks.root.glob("*.json")), [])

    def test_concurrent_same_snapshot_reuses_one_atomic_write(self):
        run_id, _, _, _, _ = self.create_run()
        barrier = Barrier(4)
        real_digest, real_save = network_storage.digest, network_storage.save
        writes = []
        def synchronized_digest(value):
            result = real_digest(value)
            barrier.wait(timeout=5)
            return result
        def tracked_save(path, value):
            writes.append(path)
            return real_save(path, value)
        with patch.object(network_storage, "digest", side_effect=synchronized_digest), patch.object(network_storage, "save", side_effect=tracked_save):
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda _: self.networks.create("Concurrent snapshot", [run_id], infer_connections=False), range(4)))
        self.assertEqual(len(writes), 1)
        self.assertEqual(len({row["network_id"] for row in results}), 1)
        self.assertEqual(len({row["created_at"] for row in results}), 1)

    def test_corrupt_cache_is_counted_without_exposing_paths_and_cannot_be_exported(self):
        network_id = str(uuid.uuid4())
        (self.networks.root / (network_id + ".json")).write_text("{broken json", encoding="utf-8")
        listing = self.client.get("/api/v1/graph-networks")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["unavailable_count"], 1)
        self.assertEqual(listing.json()["items"], [])
        for suffix in ("", "/graph", "/export"):
            response = self.client.get(f"/api/v1/graph-networks/{network_id}{suffix}")
            self.assertEqual(response.status_code, 503, response.text)
            self.assertNotIn(str(self.root), response.text)

    def test_same_material_selection_and_library_ties_are_order_independent(self):
        first, material, file, original, graph = self.create_run()
        second = self.runs.create([{"path": original, "title": "Second saved account", "locator": f"workbench:material/{material}/file/{file}"}],
                                  {"ontology": {"version": "test"}}, "test")["run_id"]
        write_json(self.runs.path(second) / "graph_input.json", graph)
        self.runs.state(self.runs.path(second), "awaiting_review", "release")
        for run_id in (first, second):
            inputs = read_json(self.runs.path(run_id) / "inputs.json")
            inputs["created_at"] = "2026-10-19T00:00:00Z"
            write_json(self.runs.path(run_id) / "inputs.json", inputs)
        forward = self.networks.create("Two accounts", [first, second], infer_connections=False)
        reverse = self.networks.create("Two accounts", [second, first], infer_connections=False)
        self.assertEqual(forward["network_id"], reverse["network_id"])
        self.assertEqual(forward["coverage"]["materials"][0]["run_ids"], sorted([first, second]))
        with patch.object(network_storage, "_all_runs", return_value=[{"run_id": first}, {"run_id": second}]):
            library_forward = self.networks.create("Stable library", [], "library", False)
        with patch.object(network_storage, "_all_runs", return_value=[{"run_id": second}, {"run_id": first}]):
            library_reverse = self.networks.create("Stable library", [], "library", False)
        self.assertEqual(library_forward["run_ids"], [max(first, second)])
        self.assertEqual(library_forward["network_id"], library_reverse["network_id"])


if __name__ == "__main__":
    unittest.main()
