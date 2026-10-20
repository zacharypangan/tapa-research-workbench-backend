"""Operational capability reporting never implies untested task completion."""
import asyncio
from contextlib import ExitStack
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, patch

from app.api import repository
from app.repository import document_representation, service_readiness
from app.repository.schemas import BuildSemanticIndexRequest
from app.repository.service_readiness import service_capabilities


class ServiceCapabilitiesTests(unittest.TestCase):
    def snapshot(self, **overrides):
        values = dict(generation_configured=True, generation_ready=True,
                      generation_detail="Text model responded.", generation_provider="ollama",
                      generation_model="local-text", embedding_configured=True,
                      embedding_ready=False, embedding_detail="HTTP 429: quota exceeded",
                      embedding_provider="gemini", embedding_model="test-embedding",
                      vision_configured=True, vision_model="local-vision")
        return service_capabilities(**{**values, **overrides})

    def test_capabilities_are_independent_and_derived_tasks_are_unverified(self):
        services = self.snapshot()
        self.assertTrue(services["generation"]["ready"])
        self.assertFalse(services["embeddings"]["ready"])
        self.assertEqual(services["embeddings"]["state"], "unavailable")
        for key in ("translation", "graph_extraction"):
            self.assertIsNone(services[key]["ready"])
            self.assertEqual(services[key]["state"], "unverified")
            self.assertEqual(services[key]["verification"], "generation_probe_only")
            self.assertTrue(services[key]["dependency_ready"])
        self.assertIsNone(services["image_description"]["ready"])
        self.assertTrue(services["keyword_search"]["ready"])

    def test_cached_vision_failure_is_a_blocker_not_a_success(self):
        service = self.snapshot(vision_failure={"error_category": "http_error", "http_status": 404})["image_description"]
        self.assertEqual(service["state"], "blocked")
        self.assertFalse(service["ready"])
        self.assertIn("HTTP 404", service["detail"])

    def test_missing_configuration_is_distinct_from_failed_model(self):
        services = self.snapshot(generation_configured=False, generation_ready=False,
                                 vision_configured=False)
        for key in ("generation", "translation", "graph_extraction", "image_description"):
            self.assertFalse(services[key]["ready"])
            self.assertEqual(services[key]["state"], "not_configured")

    def test_ocr_language_coverage_can_be_partial_without_disabling_other_languages(self):
        service = self.snapshot(ocr_runtime={"engine_version": "5.5.0", "installed": ["eng", "jpn"]})["ocr"]
        self.assertEqual(service["state"], "partial")
        self.assertTrue(service["configured"])
        self.assertFalse(service["ready"])
        self.assertEqual(service["missing_languages"], ["chi_sim", "chi_tra"])
        self.assertIn("Installed languages remain usable", service["detail"])

    def test_ocr_ready_means_runtime_not_quality_was_checked(self):
        service = self.snapshot(ocr_runtime={"engine_version": "5.5.0", "installed": list(service_readiness.OCR_LANGUAGES)})["ocr"]
        self.assertTrue(service["ready"])
        self.assertEqual(service["verification"], "runtime_check_only")
        self.assertEqual(service["missing_languages"], [])
        self.assertIn("no document page", service["detail"])

    def test_absent_ocr_keeps_native_text_search_independent(self):
        services = self.snapshot(ocr_runtime={"engine_version": "unavailable", "installed": []})
        self.assertEqual(services["ocr"]["state"], "unavailable")
        self.assertTrue(services["keyword_search"]["ready"])

    def test_ocr_runtime_checks_are_off_event_loop_cached_and_expire(self):
        service_readiness._OCR_RUNTIME_CACHE.clear()
        self.addCleanup(service_readiness._OCR_RUNTIME_CACHE.clear)
        main_thread = threading.get_ident()
        runtime = {"engine_version": "5.5.0", "installed": ["eng"]}
        def probe(languages):
            self.assertNotEqual(threading.get_ident(), main_thread)
            self.assertEqual(languages, list(service_readiness.OCR_LANGUAGES))
            return runtime
        with patch.object(document_representation, "ocr_runtime", side_effect=probe) as check:
            async def checks():
                results = await asyncio.gather(service_readiness.local_ocr_runtime(), service_readiness.local_ocr_runtime())
                self.assertEqual(results, [runtime, runtime])
                self.assertEqual(check.call_count, 1)
                service_readiness._OCR_RUNTIME_CACHE["snapshot"] = (float("-inf"), runtime)
                await service_readiness.local_ocr_runtime()
            asyncio.run(checks())
            self.assertEqual(check.call_count, 2)


class RepositoryReadinessTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for key, value in {"STORAGE_ROOT": root, "FILES_ROOT": root / "files",
                           "IMAGES_ROOT": root / "images", "DB_PATH": root / "db.sqlite"}.items():
            self.stack.enter_context(patch.object(repository, key, str(value)))
        repository._QUERY_EMBEDDINGS.clear()
        repository._READINESS_CACHE.clear()

    def test_private_bridge_readiness_avoids_competing_inference_probes(self):
        with patch.object(repository, "ai_chat_configured", return_value=True), \
                patch.object(repository, "ai_embedding_configured", return_value=True), \
                patch.object(repository, "EMBEDDING_PROVIDER", "ollama"), \
                patch("app.repository.model_worker_readiness.bridge_readiness", new=AsyncMock(return_value=(True, "Recent GPU execution verified; queue busy"))) as probe, \
                patch.object(repository, "request_embedding", new=AsyncMock(side_effect=AssertionError("no competing inference"))) as embed:
            self.assertTrue(asyncio.run(repository._probe_ollama_generation_readiness())[0])
            self.assertTrue(asyncio.run(repository._probe_embedding_readiness())[0])
            self.assertEqual(probe.await_count, 2)
            embed.assert_not_awaited()

    def test_bridge_disconnect_is_not_hidden_by_positive_provider_cache(self):
        from app.repository.model_worker_readiness import STAGING_GATEWAY
        with patch.object(repository, "OLLAMA_REPOSITORY_BASE_URL", STAGING_GATEWAY), \
                patch.object(repository, "_probe_ollama_generation_readiness", new=AsyncMock(side_effect=[(True, "ready"), (False, "disconnected")])) as probe:
            self.assertTrue(asyncio.run(repository.check_ollama_generation_readiness())[0])
            self.assertFalse(asyncio.run(repository.check_ollama_generation_readiness())[0])
            self.assertEqual(probe.await_count, 2)

    def test_self_hosted_status_never_claims_cloud_provider(self):
        with patch.object(repository, "OLLAMA_REPOSITORY_BASE_URL", "http://local-model:11434"), \
                patch.object(repository, "OLLAMA_RETRIEVAL_MODEL", "local-text"):
            self.assertNotIn("Cloud", repository.ollama_status_message(True))
            self.assertNotIn("Cloud", repository.ollama_status_message(False))

    def test_status_retains_legacy_fields_and_reports_seven_services_without_vision_request(self):
        with patch.object(repository, "check_ollama_generation_readiness", new=AsyncMock(return_value=(True, "text ready"))), \
                patch.object(repository, "check_embedding_readiness", new=AsyncMock(return_value=(False, "quota exceeded"))), \
                patch.object(repository, "ai_chat_configured", return_value=True), \
                patch.object(repository, "ai_embedding_configured", return_value=True), \
                patch.object(repository, "OLLAMA_REPOSITORY_BASE_URL", "http://local-model:11434"), \
                patch.object(repository, "local_ocr_runtime", new=AsyncMock(return_value={"engine_version": "5.5.0", "installed": ["eng"]})), \
                patch.object(repository, "request_image_caption", new=AsyncMock(side_effect=AssertionError("no vision probe"))) as vision:
            response = asyncio.run(repository.get_ai_status())
        self.assertTrue(response["provider_configured"])
        self.assertFalse(response["embedding_configured"])
        self.assertEqual(set(response["services"]), {"generation", "embeddings", "translation", "graph_extraction", "image_description", "keyword_search", "ocr"})
        self.assertEqual(response["services"]["ocr"]["state"], "partial")
        self.assertEqual(response["services"]["embeddings"]["detail"], "quota exceeded")
        self.assertNotIn("Cloud", response["status_message"])
        self.assertNotIn("http://", str(response["services"]))
        vision.assert_not_awaited()

    def test_query_embedding_cache_is_partitioned_by_endpoint_without_changing_persisted_model_key(self):
        with patch.object(repository, "EMBEDDING_PROVIDER", "ollama"), \
                patch.object(repository, "OLLAMA_EMBEDDING_MODEL", "same-model"), \
                patch.object(repository, "request_embedding", new=AsyncMock(side_effect=[[1., 0.], [0., 1.]])) as embed:
            stored_key = repository.active_embedding_index_key()
            async def queries():
                with patch.object(repository, "OLLAMA_REPOSITORY_BASE_URL", "http://server-a:11434"):
                    self.assertEqual(await repository.query_embedding_for_search("same query"), [1., 0.])
                    self.assertEqual(await repository.query_embedding_for_search("same query"), [1., 0.])
                with patch.object(repository, "OLLAMA_REPOSITORY_BASE_URL", "http://server-b:11434"):
                    self.assertEqual(await repository.query_embedding_for_search("same query"), [0., 1.])
            asyncio.run(queries())
            self.assertEqual(embed.await_count, 2)
            self.assertEqual(repository.active_embedding_index_key(), stored_key)

    def test_readiness_cache_is_partitioned_by_endpoint(self):
        with patch.object(repository, "EMBEDDING_PROVIDER", "ollama"), \
                patch.object(repository, "_probe_embedding_readiness", new=AsyncMock(side_effect=[(True, "a ready"), (False, "b offline")])) as probe:
            async def checks():
                with patch.object(repository, "OLLAMA_REPOSITORY_BASE_URL", "http://server-a:11434"):
                    self.assertTrue((await repository.check_embedding_readiness())[0])
                    self.assertTrue((await repository.check_embedding_readiness())[0])
                with patch.object(repository, "OLLAMA_REPOSITORY_BASE_URL", "http://server-b:11434"):
                    self.assertFalse((await repository.check_embedding_readiness())[0])
            asyncio.run(checks())
            self.assertEqual(probe.await_count, 2)

    def test_index_failures_return_embedding_failure_instead_of_generation_message(self):
        with patch.object(repository, "check_embedding_readiness", new=AsyncMock(return_value=(False, "HTTP 429: quota exceeded"))), \
                patch.object(repository, "embedding_status_message", return_value="Embeddings unavailable"), \
                patch.object(repository, "ollama_status_message", side_effect=AssertionError("wrong service")):
            for endpoint in (repository.build_semantic_index, repository.build_image_index):
                result = asyncio.run(endpoint(BuildSemanticIndexRequest()))
                self.assertFalse(result["provider_configured"])
                self.assertEqual(result["status_detail"], "HTTP 429: quota exceeded")
                self.assertEqual(result["message"], "Embeddings unavailable")


if __name__ == "__main__":
    unittest.main()
