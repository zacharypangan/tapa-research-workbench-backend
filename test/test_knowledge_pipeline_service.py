import asyncio
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path

from app.repository.knowledge_pipeline.adapters import ingest_document
from app.repository.knowledge_pipeline.processing import process_response
from app.repository.knowledge_pipeline.service import ResearchPipeline, batches_for, blocking_stage, validate_config
from app.repository.knowledge_pipeline.store import RunStore, digest, file_hash, read_json, write_json


ONTOLOGY = {"version": "test-1", "entity_types": {"ToolForm": {}, "MaterialKind": {}},
            "predicates": {"processes_material": {"subject_types": ["ToolForm"], "object_types": ["MaterialKind"], "requires_object": True},
                           "aligns_reported_concept_with": {"requires_object": True}}}


def response_for(units):
    evidence = [{"unit_id": units[0]["unit_id"], "quote": units[0]["text"]}]
    return {"entities": [{"local_id": "tool", "entity_type": "ToolForm", "label_original": "stone beater", "evidence": evidence},
                         {"local_id": "material", "entity_type": "MaterialKind", "label_original": "barkcloth", "evidence": evidence}],
            "assertions": [{"local_id": "claim", "subject": "tool", "predicate": "processes_material", "object": "material",
                            "scope": {}, "polarity": "positive", "epistemic_kind": "source_report", "evidence": evidence,
                            "rationale": "Synthetic fixture explicitly reports this relationship."}]}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.document = self.root / "document.txt"
        self.document.write_text("A stone beater processes barkcloth.")
        self.store = RunStore(self.root / "runs", "project-a")
        self.calls = {"model": 0, "web": 0}

        async def provider(units, ontology):
            self.calls["model"] += 1
            response = response_for(units)
            return {**process_response(response, units, ontology), "provider_response": response,
                    "provider_metadata": {"method": "synthetic_test_provider", "response_sha256": digest(response)}}

        def collector(frontier, destination, options):
            self.calls["web"] += 1
            fixture = self.root / "web-fixture.txt"
            fixture.write_text("A stone beater processes barkcloth during tapa manufacture.")
            result = ingest_document(fixture, "WEB_FIXTURE", "Synthetic collected page", destination, origin="web")
            result["source"]["seed_ids"] = frontier[0]["seed_ids"]
            result["source"]["source_locator"] = frontier[0]["url"]
            for unit in result["units"]:
                unit["seed_ids"] = frontier[0]["seed_ids"]
            return {"sources": [result["source"]], "units": result["units"], "attempts": [{"status": "collected", "url": frontier[0]["url"]}]}

        self.pipeline = ResearchPipeline(self.store, provider, collector)

    def tearDown(self):
        self.temp.cleanup()

    def create(self, mode="ollama", enabled=True, key=None):
        return self.store.create([{"path": self.document, "title": "Synthetic primary document"}],
                                 validate_config({"ontology": ONTOLOGY, "processing": {"mode": mode},
                                                  "augmentation": {"enabled": enabled}}), "researcher", key)["run_id"]

    def advance(self, run_id):
        return asyncio.run(self.pipeline.run(run_id))

    def test_document_seeded_web_review_release_and_checkpoint_replay(self):
        run_id = self.create()
        state = self.advance(run_id)
        self.assertEqual(state["status"], "awaiting_frontier")
        path = self.store.path(run_id)
        seeds = read_json(path / "seeds.json")
        self.assertEqual(len(seeds), 2)
        self.pipeline.submit_frontier(run_id, [{"url": "https://example.org/tapa", "seed_ids": [s["seed_id"] for s in seeds]}], "researcher")
        state = self.advance(run_id)
        self.assertEqual(state["status"], "awaiting_review")
        graph = read_json(path / "graph_input.json")
        self.assertEqual([len(graph[k]) for k in ("sources", "entities", "assertions", "alignments")], [2, 4, 2, 2])
        for alignment in graph["alignments"]:
            evidence = {e["evidence_id"]: e for e in graph["evidence"]}
            self.assertEqual(len({evidence[e]["source_id"] for e in alignment["evidence_ids"]}), 2)
            self.assertIn(alignment["subject_id"], graph["sources"][1]["seed_ids"])
        self.assertEqual(self.calls, {"model": 2, "web": 1})
        self.advance(run_id)
        self.assertEqual(self.calls, {"model": 2, "web": 1})
        preview = self.pipeline.release(run_id, "reviewer", allow_pending=True)
        self.assertEqual(preview["status"], "awaiting_review")
        decisions = [{"candidate_id": row[key], "decision": "accept", "rationale": "Fixture inclusion check; scientific review remains pending"}
                     for field, key in (("entities", "entity_id"), ("assertions", "assertion_id"), ("alignments", "alignment_id")) for row in graph[field]]
        self.pipeline.review(run_id, decisions, "reviewer")
        released = self.pipeline.release(run_id, "reviewer")
        self.assertEqual(released["status"], "released")
        release = path / released["release_path"]
        compiled = read_json(release / "graph.json")
        self.assertEqual(len(compiled["alignments"]), 2)
        self.assertIn("kp:scopeJSON", (release / "graph.ttl").read_text())
        manifest = read_json(release / "manifest.json")
        self.assertEqual(set(manifest["artifact_sha256"]) | {"manifest.json"}, {file.name for file in release.iterdir()})
        for name, sha in manifest["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((release / name).read_bytes()).hexdigest(), sha)
        self.assertEqual(self.advance(run_id)["status"], "released")
        self.assertTrue(list((path / "provider_responses").glob("*.json")))

    def test_manual_annotations_and_immutable_response(self):
        run_id = self.create(mode="manual", enabled=False)
        state = self.advance(run_id)
        self.assertEqual(state["status"], "awaiting_annotations")
        batch = read_json(self.store.path(run_id) / "primary_batches.json")[0]
        self.pipeline.submit_responses(run_id, {batch["batch_id"]: response_for(batch["units"])})
        with self.assertRaises(ValueError):
            self.pipeline.submit_responses(run_id, {batch["batch_id"]: {"entities": [], "assertions": []}})
        self.assertEqual(self.advance(run_id)["status"], "awaiting_review")
        self.assertEqual(self.calls, {"model": 0, "web": 0})

    def test_derived_retry_revalidates_completed_primary_response_without_provider_work(self):
        parent = self.create(enabled=False)
        self.assertEqual(self.advance(parent)["status"], "awaiting_review")
        child = self.store.derive(parent, "researcher")["run_id"]
        self.assertEqual(self.advance(child)["status"], "awaiting_review")
        self.assertEqual(self.calls["model"], 1)
        parent_path, child_path = self.store.path(parent), self.store.path(child)
        batch = read_json(parent_path / "primary_batches.json")[0]["batch_id"]
        previous = read_json(parent_path / "processed" / f"{batch}.json")
        current = read_json(child_path / "processed" / f"{batch}.json")
        for key in ("entities", "assertions", "evidence"):
            self.assertEqual(previous[key], current[key])
        metadata = current["processing_metadata"]
        self.assertEqual(metadata["method"], previous["processing_metadata"]["method"])
        self.assertEqual(metadata["response_reuse"]["reused_from_run_id"], parent)
        self.assertEqual(metadata["response_reuse"]["parent_response_sha256"], file_hash(parent_path / "provider_responses" / f"{batch}.json"))
        self.assertEqual(metadata["response_reuse"]["revalidated_implementation_sha256"],
                         {key: file_hash(Path(__file__).parents[1] / "app/repository/knowledge_pipeline" / key)
                          for key in ("processing.py", "service.py")})
        self.assertEqual(read_json(parent_path / "provider_responses" / f"{batch}.json"),
                         read_json(child_path / "provider_responses" / f"{batch}.json"))

    def test_derived_retry_does_not_reuse_rejected_primary_batch(self):
        valid_provider = self.pipeline.provider
        async def invalid_provider(units, ontology):
            self.calls["model"] += 1
            response = response_for(units)
            response["entities"][0]["label_original"] = "unsupported hammer"
            return {**process_response(response, units, ontology), "provider_response": response}
        self.pipeline.provider = invalid_provider
        parent = self.create(enabled=False)
        self.assertEqual(self.advance(parent)["status"], "awaiting_annotations")
        self.pipeline.provider = valid_provider
        child = self.store.derive(parent, "researcher")["run_id"]
        self.assertEqual(self.advance(child)["status"], "awaiting_review")
        self.assertEqual(self.calls["model"], 2)
        checkpoint = next((self.store.path(child) / "processed").glob("*.json"))
        self.assertNotIn("response_reuse", read_json(checkpoint)["processing_metadata"])

    def test_derived_retry_does_not_reuse_same_batch_with_changed_input_identity(self):
        parent = self.create(enabled=False)
        self.advance(parent)
        child = self.store.derive(parent, "researcher", source_metadata={"document.txt": {"filename": "changed-original-name.txt"}})["run_id"]
        self.assertEqual(self.advance(child)["status"], "awaiting_review")
        self.assertEqual(read_json(self.store.path(parent) / "primary_batches.json"), read_json(self.store.path(child) / "primary_batches.json"))
        self.assertEqual(self.calls["model"], 2)

    def test_derived_retry_falls_back_when_fresh_validation_changes_candidates(self):
        parent = self.create(enabled=False)
        self.advance(parent)
        checkpoint = next((self.store.path(parent) / "processed").glob("*.json"))
        previous = read_json(checkpoint)
        previous["entities"][0]["review_status"] = "unexpected_legacy_status"
        write_json(checkpoint, previous)
        child = self.store.derive(parent, "researcher")["run_id"]
        self.assertEqual(self.advance(child)["status"], "awaiting_review")
        self.assertEqual(self.calls["model"], 2)
        current = read_json(self.store.path(child) / "processed" / checkpoint.name)
        self.assertNotIn("response_reuse", current["processing_metadata"])
        self.assertTrue(all(entity["review_status"] == "needs_specialist_review" for entity in current["entities"]))

    def test_idempotency_snapshot_and_tenant_isolation(self):
        run_id = self.create(key="request-1")
        self.assertEqual(self.create(key="request-1"), run_id)
        with self.assertRaises(FileNotFoundError):
            RunStore(self.root / "runs", "project-b").describe(run_id)
        with self.assertRaises(FileNotFoundError):
            self.store.artifact(run_id, "../idempotency.json")
        self.document.write_text("Changed source")
        with self.assertRaises(ValueError):
            self.create(key="request-1")
        self.assertEqual(read_json(self.store.path(run_id) / "inputs.json")["documents"][0]["sha256"], hashlib.sha256(b"A stone beater processes barkcloth.").hexdigest())

    def test_unbound_and_private_frontiers_are_rejected(self):
        run_id = self.create()
        self.advance(run_id)
        seed = read_json(self.store.path(run_id) / "seeds.json")[0]["seed_id"]
        for url, seed_ids in [("http://127.0.0.1/internal", [seed]), ("https://example.org/", ["foreign-seed"]), ("http://localhost/", [seed])]:
            with self.assertRaises(ValueError):
                self.pipeline.submit_frontier(run_id, [{"url": url, "seed_ids": seed_ids}], "researcher")
        self.assertEqual(self.calls["web"], 0)

    def test_long_units_are_not_silently_truncated(self):
        text = "x" * 2501
        batches = batches_for([{"unit_id": "unit", "source_id": "doc", "locator": {"page": 1}, "text": text}], 1000)
        self.assertEqual("".join(b["units"][0]["text"] for b in batches), text)
        self.assertEqual(batches[-1]["units"][0]["locator"]["start_char"], 2000)

    def test_malformed_options_fail_validation(self):
        for config in ({"processing": None}, {"processing": {"mode": []}}, {"augmentation": {"enabled": "false"}},
                       {"augmentation": {"max_pages": "5"}}, {"processing": {"batch_chars": True}}):
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_no_primary_candidates_does_not_deadlock_frontier(self):
        async def empty_provider(units, ontology):
            return process_response({"entities": [], "assertions": []}, units, ontology)
        self.pipeline.provider = empty_provider
        run_id = self.create()
        state = self.advance(run_id)
        self.assertEqual(state["status"], "awaiting_review")
        self.assertEqual(state["entity_count"], 0)
        self.assertEqual(self.calls["web"], 0)

    def test_cancellation_waits_for_file_writer_before_unlock(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def writer():
            entered.set()
            release.wait(timeout=2)
            finished.set()
        async def scenario():
            task = asyncio.create_task(blocking_stage(writer))
            await asyncio.to_thread(entered.wait, 1)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(finished.is_set())
        asyncio.run(scenario())

    def test_failed_provider_can_resume_with_submitted_annotations(self):
        async def unavailable(units, ontology):
            raise RuntimeError("Synthetic provider unavailable")
        self.pipeline.provider = unavailable
        run_id = self.create(enabled=False)
        with self.assertRaises(RuntimeError):
            self.advance(run_id)
        batch = read_json(self.store.path(run_id) / "primary_batches.json")[0]
        self.pipeline.submit_responses(run_id, {batch["batch_id"]: response_for(batch["units"])})
        self.assertEqual(self.advance(run_id)["status"], "awaiting_review")

    def test_invalid_provider_batch_pauses_without_empty_graph_and_accepts_correction(self):
        invalid = {"entities": [{"local_id": "E1", "label": "stone beater", "type": "ToolForm"}],
                   "features": [], "assertions": []}
        async def wrong_shape(units, ontology):
            self.calls["model"] += 1
            return {**process_response(invalid, units, ontology), "provider_response": invalid,
                    "provider_metadata": {"method": "synthetic_invalid_provider"}}
        self.pipeline.provider = wrong_shape
        run_id = self.create(enabled=False)
        state = self.advance(run_id)
        path = self.store.path(run_id)
        batch = read_json(path / "primary_batches.json")[0]
        identifier = batch["batch_id"]
        self.assertEqual(state["status"], "awaiting_annotations")
        self.assertEqual(state["rejected_batch_ids"], [identifier])
        self.assertFalse((path / "primary.json").exists())
        self.assertFalse((path / "graph_input.json").exists())
        self.assertFalse((path / "processed" / f"{identifier}.json").exists())
        audit = read_json(path / "rejected_batches" / f"{identifier}.json")
        self.assertEqual(audit["response"], invalid)
        self.assertEqual(audit["validation"]["rejections"][0]["item_kind"], "batch")
        self.assertEqual(self.advance(run_id)["status"], "awaiting_annotations")
        self.assertEqual(self.calls["model"], 1)
        with self.assertRaisesRegex(ValueError, "Annotations could not be validated"):
            self.pipeline.submit_responses(run_id, {identifier: invalid})
        self.assertFalse((path / "responses.json").exists())
        self.pipeline.submit_responses(run_id, {identifier: response_for(batch["units"])})
        self.assertEqual(self.advance(run_id)["status"], "awaiting_review")
        graph = read_json(path / "graph_input.json")
        self.assertEqual(len(graph["entities"]), 2)
        self.assertEqual(len(graph["assertions"]), 1)
        self.assertEqual(self.calls["model"], 1)
        self.assertEqual(read_json(path / "rejected_batches" / f"{identifier}.json")["response"], invalid)

    def test_rejected_web_batch_pauses_without_losing_valid_primary_checkpoint(self):
        valid_provider = self.pipeline.provider
        async def partly_invalid(units, ontology):
            if units[0]["source_id"] != "WEB_FIXTURE":
                return await valid_provider(units, ontology)
            invalid = response_for(units)
            invalid["entities"][0]["label_original"] = "unsupported hammer"
            return {**process_response(invalid, units, ontology), "provider_response": invalid}
        self.pipeline.provider = partly_invalid
        run_id = self.create()
        self.assertEqual(self.advance(run_id)["status"], "awaiting_frontier")
        path = self.store.path(run_id)
        primary = (path / "primary.json").read_bytes()
        seed = read_json(path / "seeds.json")[0]["seed_id"]
        self.pipeline.submit_frontier(run_id, [{"url": "https://example.org/fixture", "seed_ids": [seed]}], "fixture")
        state = self.advance(run_id)
        self.assertEqual(state["status"], "awaiting_annotations")
        self.assertEqual(state["stage"], "augmented")
        self.assertEqual((path / "primary.json").read_bytes(), primary)
        self.assertFalse((path / "graph_input.json").exists())
        batch = read_json(path / "augmented_batches.json")[0]
        self.pipeline.submit_responses(run_id, {batch["batch_id"]: response_for(batch["units"])})
        self.assertEqual(self.advance(run_id)["status"], "awaiting_review")
        self.assertEqual((path / "primary.json").read_bytes(), primary)


if __name__ == "__main__":
    unittest.main()
