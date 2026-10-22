"""Provider dispatch and immutable routing checks; no live model or network."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.api import repository
from app.repository import document_graph_candidates as candidates
from app.repository import document_graph_runtime as runtime
from app.repository.knowledge_pipeline import jobs
from app.repository.knowledge_pipeline.store import RunStore, digest, read_json


class DocumentGraphRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.legacy_provider = AsyncMock(return_value={"legacy": True})
        self.staged_provider = AsyncMock(return_value={"staged": True})
        self.pipeline = SimpleNamespace(provider=self.legacy_provider)
        self.store = object()
        self.units = [{"unit_id": "source:unit:1", "text": "Stone beaters process barkcloth."}]
        self.base = "https://provider.example/api"
        self.patchers = [
            patch.object(jobs, "configured_pipeline", return_value=self.pipeline),
            patch.object(repository, "OLLAMA_RETRIEVAL_MODEL", "test-model:1"),
            patch.object(repository, "effective_ollama_base_url", return_value=self.base),
            patch.object(repository, "ollama_headers", return_value={"X-Test": "local-fixture"}),
            patch.object(candidates, "provider", self.staged_provider),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.identity = runtime.candidate_identity()
        self.execution = runtime.execution_identity()

    def ontology(self):
        return {"version": "test", "extraction_guidance": {
            "candidate_provider": copy.deepcopy(self.identity),
            "execution_identity": copy.deepcopy(self.execution)}}

    async def test_legacy_runs_keep_the_original_provider_and_input_protocol(self):
        configured = runtime.configured_pipeline(self.store)
        for ontology in ({"version": "legacy"}, {"version": "legacy", "extraction_guidance": {"focus": "old guidance"}}):
            before = copy.deepcopy(ontology)
            self.assertEqual(await configured.provider(self.units, ontology), {"legacy": True})
            self.assertEqual(ontology, before)
        self.assertEqual(self.legacy_provider.await_count, 2)
        self.staged_provider.assert_not_awaited()

    async def test_matching_pinned_provider_receives_exact_units_ontology_and_routing(self):
        ontology = self.ontology()
        before = copy.deepcopy(ontology)
        configured = runtime.configured_pipeline(self.store)
        self.assertIs(configured, self.pipeline)
        self.assertEqual(await configured.provider(self.units, ontology), {"staged": True})
        self.staged_provider.assert_awaited_once_with(
            self.units, ontology, self.base, "test-model:1", {"X-Test": "local-fixture"})
        self.assertEqual(ontology, before)
        self.legacy_provider.assert_not_awaited()

    async def test_candidate_code_or_limit_drift_fails_closed_without_legacy_fallback(self):
        configured = runtime.configured_pipeline(self.store)
        for key, value in (("code_sha256", "different-code"), ("limits", {"max_calls": 999}), ("adapter_version", "future-version")):
            with self.subTest(key=key):
                ontology = self.ontology()
                ontology["extraction_guidance"]["candidate_provider"][key] = value
                with self.assertRaisesRegex(ValueError, "Candidate provider changed"):
                    await configured.provider(self.units, ontology)
        self.staged_provider.assert_not_awaited()
        self.legacy_provider.assert_not_awaited()

    async def test_model_or_provider_routing_drift_fails_closed(self):
        ontology = self.ontology()
        configured = runtime.configured_pipeline(self.store)
        with patch.object(repository, "OLLAMA_RETRIEVAL_MODEL", "different-model:2"):
            with self.assertRaisesRegex(ValueError, "model routing changed"):
                await configured.provider(self.units, ontology)
        with patch.object(repository, "effective_ollama_base_url", return_value="https://other-provider.example/api"):
            with self.assertRaisesRegex(ValueError, "model routing changed"):
                await configured.provider(self.units, ontology)
        self.staged_provider.assert_not_awaited()
        self.legacy_provider.assert_not_awaited()

    async def test_missing_or_changed_execution_hash_fails_closed(self):
        configured = runtime.configured_pipeline(self.store)
        for replacement in (None, {}, {**self.execution, "runtime_code_sha256": "old-runtime"}, {**self.execution, "provider_sha256": "old-provider"}):
            with self.subTest(identity=replacement):
                ontology = self.ontology()
                ontology["extraction_guidance"]["execution_identity"] = replacement
                with self.assertRaisesRegex(ValueError, "model routing changed"):
                    await configured.provider(self.units, ontology)
        self.staged_provider.assert_not_awaited()
        self.legacy_provider.assert_not_awaited()

    async def test_staged_provider_failure_is_never_replayed_through_legacy_provider(self):
        self.staged_provider.side_effect = RuntimeError("synthetic provider failure")
        configured = runtime.configured_pipeline(self.store)
        with self.assertRaisesRegex(RuntimeError, "synthetic provider failure"):
            await configured.provider(self.units, self.ontology())
        self.legacy_provider.assert_not_awaited()

    async def test_quota_failure_is_checkpointed_before_further_batch_calls_pause(self):
        self.staged_provider.return_value = {"provider_attempts": [{"http_status": 429}], "provider_metadata": {}}
        configured = runtime.configured_pipeline(self.store)
        result = await configured.provider(self.units, self.ontology())
        self.assertEqual(result["provider_metadata"]["pause_after_checkpoint"], "provider_unavailable")
        with self.assertRaisesRegex(RuntimeError, "saved checkpoints remain resumable"):
            await configured.provider(self.units, self.ontology())
        self.staged_provider.assert_awaited_once()

    async def test_storage_above_old_guard_but_below_batch_headroom_stops_inference(self):
        configured = runtime.configured_pipeline(self.store)
        with tempfile.TemporaryDirectory() as directory, patch.object(repository, "STORAGE_ROOT", directory), \
                patch.object(runtime.shutil, "disk_usage", return_value=SimpleNamespace(free=64 * 1024 * 1024)):
            with self.assertRaisesRegex(OSError, "checkpoint storage is low"):
                await configured.provider(self.units, self.ontology())
        self.staged_provider.assert_not_awaited()

    async def test_storage_headroom_boundary_prevents_model_call_below_but_allows_at_limit(self):
        configured = runtime.configured_pipeline(self.store)
        required = runtime.checkpoint_headroom_bytes()
        with tempfile.TemporaryDirectory() as directory, patch.object(repository, "STORAGE_ROOT", directory), \
                patch.object(runtime.shutil, "disk_usage") as usage:
            usage.return_value = SimpleNamespace(free=required - 1)
            with self.assertRaisesRegex(OSError, "checkpoint storage is low"):
                await configured.provider(self.units, self.ontology())
            self.staged_provider.assert_not_awaited()
            usage.return_value = SimpleNamespace(free=required)
            self.assertEqual(await configured.provider(self.units, self.ontology()), {"staged": True})
        self.staged_provider.assert_awaited_once()

    def test_storage_headroom_tracks_call_and_response_limits_with_minimum_floor(self):
        with patch.dict(candidates.LIMITS, {"max_calls": 20, "response_bytes": 2 * 1024 * 1024,
                                           "prompt_bytes": 11000}):
            self.assertEqual(runtime.checkpoint_headroom_bytes(),
                             2 * 20 * (4 * 2 * 1024 * 1024 + 2 * 11000) + 16 * 1024 * 1024)
        with patch.dict(candidates.LIMITS, {"max_calls": 1, "response_bytes": 1, "prompt_bytes": 1}):
            self.assertEqual(runtime.checkpoint_headroom_bytes(), 128 * 1024 * 1024)

    async def test_equivalent_trailing_slash_does_not_change_provider_identity(self):
        ontology = self.ontology()
        configured = runtime.configured_pipeline(self.store)
        with patch.object(repository, "effective_ollama_base_url", return_value=self.base + "/"):
            self.assertEqual(runtime.execution_identity(), self.execution)
            self.assertEqual(await configured.provider(self.units, ontology), {"staged": True})

    def test_identity_hashes_runtime_and_route_without_exposing_provider_endpoint(self):
        self.assertEqual(self.identity["version"], self.identity["adapter_version"])
        self.assertTrue(self.identity["requires_review"])
        self.assertEqual(self.execution["provider_sha256"], digest(self.base))
        self.assertEqual(self.execution["runtime_code_sha256"], hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest())
        self.assertNotIn("https://", str(self.execution))

    def test_unconfigured_legacy_provider_is_preserved_without_implicit_model_call(self):
        self.pipeline.provider = None
        configured = runtime.configured_pipeline(self.store)
        self.assertIs(configured, self.pipeline)
        self.assertIsNone(configured.provider)
        self.staged_provider.assert_not_called()


class DocumentGraphSchedulerIdentityTests(unittest.TestCase):
    def test_existing_durable_scheduler_is_reused_and_factory_is_wrapped_once(self):
        scheduler = SimpleNamespace(pipeline_factory=jobs.configured_pipeline)
        with patch.object(jobs, "get_scheduler", return_value=scheduler) as existing:
            self.assertIs(runtime.get_scheduler(), scheduler)
            self.assertIs(scheduler.pipeline_factory, runtime.configured_pipeline)
            self.assertIs(runtime.get_scheduler(), scheduler)
            self.assertEqual(existing.call_count, 2)

    def test_custom_scheduler_factory_is_not_overwritten(self):
        custom = object()
        scheduler = SimpleNamespace(pipeline_factory=custom)
        with patch.object(jobs, "get_scheduler", return_value=scheduler):
            self.assertIs(runtime.get_scheduler().pipeline_factory, custom)

    def test_pipeline_identity_matches_the_exact_files_pinned_by_run_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "source.txt"
            original.write_text("A source passage.", encoding="utf-8")
            store = RunStore(root / "runs", "local-test")
            run = store.create([{"path": original, "title": "Source", "locator": "test:source"}],
                               {"ontology": {"version": "test"}}, "test")
            inputs = read_json(store.path(run["run_id"]) / "inputs.json")
            self.assertEqual(runtime.pipeline_identity(), digest(inputs["implementation_sha256"]))
