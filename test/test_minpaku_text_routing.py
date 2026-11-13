"""Production text cutover uses an independent route and truthful admission."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import HTTPException

from app.api import repository, translations
from app.repository import graph_generation_profiles as profiles, model_worker_readiness as worker
from app.repository.knowledge_pipeline import jobs
from app.repository.schemas import AskCorpusRequest


class MinpakuTextRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        for name, value in {
            "MINPAKU_TEXT_ENABLED": True, "MINPAKU_GATEWAY_TOKEN": "gateway-test-token-" * 2,
            "OLLAMA_REPOSITORY_API_KEY": "cloud-test-token", "OLLAMA_REPOSITORY_BASE_URL": "",
            "OLLAMA_RETRIEVAL_MODEL": worker.MINPAKU_TEXT_MODEL, "OLLAMA_VISION_MODEL": "llava",
            "EMBEDDING_PROVIDER": "gemini", "STORAGE_ROOT": str(directory),
            "DB_PATH": str(directory / "repo.sqlite"), "FILES_ROOT": str(directory / "files"),
            "IMAGES_ROOT": str(directory / "images"),
        }.items():
            stack.enter_context(patch.object(repository, name, value))

    async def test_idle_worker_enables_first_request_without_claiming_verified_inference(self):
        request = AskCorpusRequest(question="What tool was used to make barkcloth?")
        with patch.object(repository, "minpaku_text_readiness", new=AsyncMock(return_value=worker.WorkerReadiness(True, False, "Connected; inference unverified."))), \
                patch.object(repository, "check_embedding_readiness", new=AsyncMock(return_value=(True, "ready"))), \
                patch.object(repository, "review_retrieval", new=AsyncMock(return_value=({"results": [{"citation": "source"}], "related_observations": []}, {"image_results": []}))), \
                patch.object(repository, "request_evidence_answer", new=AsyncMock(return_value="A stone tool [1].")) as answer:
            status = await repository.get_ai_status()
            self.assertTrue(status["chat_configured"])
            self.assertEqual(status["services"]["generation"]["state"], "unverified")
            self.assertEqual(status["services"]["generation"]["provider"], "minpaku")
            self.assertEqual(status["services"]["image_description"]["provider"], "ollama-cloud")
            self.assertEqual(status["services"]["image_description"]["model"], "llava")
            self.assertEqual(status["services"]["embeddings"]["provider"], "gemini")
            self.assertEqual((await repository.ask_corpus(request))["answer_origin"], "model_generated")
            answer.assert_awaited_once()

    async def test_disconnected_worker_cannot_start_assisted_review(self):
        with patch.object(repository, "minpaku_text_readiness", new=AsyncMock(return_value=worker.WorkerReadiness(False, False, "Disconnected"))), \
                patch.object(repository, "review_retrieval", new=AsyncMock()) as retrieve:
            with self.assertRaises(HTTPException) as caught:
                await repository.ask_corpus(AskCorpusRequest(question="What tool was used?"))
            self.assertEqual(caught.exception.status_code, 503)
            retrieve.assert_not_called()

    async def test_text_uses_only_gateway_credential_and_ollama_embedding_keeps_its_old_route(self):
        captured = []
        transport = httpx.MockTransport(lambda request: captured.append(request) or httpx.Response(
            200, json={"message": {"content": "answer"}, "embeddings": [[1.0, 0.0]]}))
        client = httpx.AsyncClient(transport=transport)
        with patch.object(repository.httpx, "AsyncClient", side_effect=[client, httpx.AsyncClient(transport=transport)]):
            self.assertEqual(await repository.request_evidence_answer("Test?", [{"material_title": "Fixture", "content_text": "Example"}]), "answer")
            self.assertEqual(await repository.request_ollama_embedding("Fixture"), [1.0, 0.0])
        self.assertEqual(str(captured[0].url), worker.PRODUCTION_GATEWAY + "/api/chat")
        self.assertEqual(captured[0].headers["Authorization"], f"Bearer {repository.MINPAKU_GATEWAY_TOKEN}")
        self.assertEqual(str(captured[1].url), "https://ollama.com/api/embed")
        self.assertEqual(captured[1].headers["Authorization"], "Bearer cloud-test-token")

    async def test_graph_switch_does_not_mix_an_unpinned_run_or_switch_back_a_pinned_one(self):
        provider = AsyncMock(return_value={"entities": []})
        config = profiles.prepare_generation_config({"processing": {"mode": "ollama"}})
        self.assertEqual(config["ontology"]["extraction_guidance"]["text_route"]["model"], worker.MINPAKU_TEXT_MODEL)
        with patch.object(jobs, "configured_pipeline", side_effect=lambda store: SimpleNamespace(provider=provider)):
            routed = profiles.configured_pipeline(object()).provider
            with self.assertRaisesRegex(ValueError, "Graph text routing changed"):
                await routed([], {})
            provider.assert_not_called()
            await routed([], config["ontology"])
            provider.assert_awaited_once()
            with patch.object(repository, "MINPAKU_TEXT_ENABLED", False):
                with self.assertRaisesRegex(ValueError, "Graph text routing changed"):
                    await routed([], config["ontology"])

    async def test_old_translation_retry_creates_a_new_job_and_preserves_old_artifacts(self):
        scheduler = Mock()
        scheduler.get.return_value = {"status": "partial", "model": "gpt-oss:120b", "material_id": "source", "file_id": "file"}
        scheduler.enqueue.return_value = {"job_id": "new"}
        with patch.object(translations, "get_translation_scheduler", return_value=scheduler):
            self.assertEqual(await translations.retry_translation("old"), {"job_id": "new"})
        scheduler.enqueue.assert_called_once_with("source", "file")
        scheduler.retry.assert_not_called()
        scheduler.update.assert_not_called()

    async def test_private_health_rejects_stale_heartbeat_and_accepts_recent_success_separately(self):
        now = time.time()
        model = {"worker_fresh": True, "last_seen_at": now - 5, "ready": True,
                 "last_success_at": now - 10, "last_success_operation": "chat"}
        health = {"checked_at": now, "worker_max_age_seconds": 35, "success_max_age_seconds": 300,
                  "queued_count": 0, "running_count": 0, "version": worker.READINESS_VERSION,
                  "models": {worker.MINPAKU_TEXT_MODEL: model}}
        for updates, expected in [({}, (True, True)), ({"last_success_at": now - 301}, (True, False)),
                                  ({"last_seen_at": now - 36}, (False, False))]:
            model.update(updates)
            transport = httpx.MockTransport(lambda request: httpx.Response(200, json=health))
            client = httpx.AsyncClient(transport=transport)
            with patch.object(worker.httpx, "AsyncClient", return_value=client):
                self.assertEqual((await worker.minpaku_text_readiness({}))[:2], expected)
