"""English copies retain immutable originals and retry only unfinished units."""
import asyncio
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.repository import translations as module
from app.repository.processing_jobs import save


class TranslationFixture:
    def fixture(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.inputs = {"material_id": "source", "files": [{"file_id": "paper", "sha256": "a" * 64,
                       "filename": "中文.pdf"}], "language": "zh", "source_url": None}
        self.stack.enter_context(patch.object(module, "material_inputs", side_effect=lambda _: json.loads(json.dumps(self.inputs))))
        self.stack.enter_context(patch.object(module, "configured_model", return_value="gpt-oss:120b"))
        self.provider = AsyncMock(return_value={"translation_text": "Translated source text", "response_sha256": "b" * 64})
        self.scheduler = module.TranslationScheduler(self.root, self.provider)

    def job_with_plan(self, count=2, complete=True):
        job = self.scheduler.enqueue("source")
        units = []
        for index in range(count):
            text = f"原文{index}"
            units.append({"unit_id": f"unit{index}", "file_id": "paper", "filename": "中文.pdf", "locator": f"page:{index+1}",
                          "page_index": index+1, "start": 0, "end": len(text), "original_text": text,
                          "original_sha256": hashlib.sha256(text.encode()).hexdigest(), "provenance": {"model": job["model"]}})
        save(self.scheduler.directory(job["job_id"]) / "units.json", {"units": units, "sources_complete": complete, "warnings": []})
        return job


class TranslationJobsTests(TranslationFixture, IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture()

    async def test_complete_keeps_originals_and_retry_does_not_repeat_provider(self):
        job = self.job_with_plan()
        await self.scheduler.process(job["job_id"])
        artifact = self.scheduler.artifact(job["job_id"])
        self.assertEqual(self.scheduler.get(job["job_id"])["status"], "done")
        self.assertEqual(artifact["coverage"], {"total": 2, "done": 2, "failed": 0, "pending": 0})
        self.assertEqual(artifact["units"][0]["original_text"], "原文0")
        self.assertTrue(artifact["review_required"])
        self.scheduler.retry(job["job_id"])
        await self.scheduler.process(job["job_id"])
        self.assertEqual(self.provider.await_count, 2)

    async def test_permanent_failure_stops_and_retry_preserves_successful_unit(self):
        job = self.job_with_plan(3)
        self.provider.side_effect = [{"translation_text": "First translated page"}, module.TranslationError("Provider HTTP 404", http_status=404)]
        await self.scheduler.process(job["job_id"])
        self.assertEqual(self.provider.await_count, 2)
        self.assertEqual(self.scheduler.get(job["job_id"])["status"], "partial")
        artifact = self.scheduler.artifact(job["job_id"])
        self.assertEqual(artifact["coverage"], {"total": 3, "done": 1, "failed": 1, "pending": 1})
        first = (self.scheduler.directory(job["job_id"]) / "unit0.json").read_bytes()
        self.provider.side_effect = None
        self.scheduler.retry(job["job_id"])
        await self.scheduler.process(job["job_id"])
        self.assertEqual(self.provider.await_count, 4)
        self.assertEqual((self.scheduler.directory(job["job_id"]) / "unit0.json").read_bytes(), first)
        self.assertEqual(self.scheduler.get(job["job_id"])["status"], "done")

    async def test_source_changes_supersede_before_provider_and_retry_gets_new_identity(self):
        job = self.job_with_plan()
        self.inputs["files"][0]["sha256"] = "c" * 64
        await self.scheduler.process(job["job_id"])
        self.provider.assert_not_awaited()
        self.assertEqual(self.scheduler.get(job["job_id"])["status"], "superseded")
        replacement = self.scheduler.retry(job["job_id"])
        self.assertNotEqual(replacement["job_id"], job["job_id"])
        self.assertEqual(self.scheduler.get(job["job_id"])["replacement_job_id"], replacement["job_id"])

    async def test_source_changes_during_call_preserve_historical_copy_but_never_current_done(self):
        job = self.job_with_plan(1)
        async def provider(*_):
            self.inputs["files"][0]["sha256"] = "d" * 64
            return {"translation_text": "Historical translated page"}
        self.provider.side_effect = provider
        await self.scheduler.process(job["job_id"])
        self.assertEqual(self.scheduler.get(job["job_id"])["status"], "superseded")
        self.assertTrue(self.scheduler.is_stale(self.scheduler.get(job["job_id"])))
        self.assertEqual(self.scheduler.artifact(job["job_id"])["coverage"]["done"], 1)

    async def test_transient_failures_are_bounded_and_partial_ocr_never_complete(self):
        job = self.job_with_plan(1, complete=False)
        self.provider.side_effect = module.TranslationError("Temporarily unavailable", retryable=True)
        with patch.object(module.asyncio, "sleep", new=AsyncMock()):
            await self.scheduler.process(job["job_id"])
        self.assertEqual(self.provider.await_count, 2)
        self.assertEqual(self.scheduler.get(job["job_id"])["status"], "error")
        self.provider.side_effect = None
        self.scheduler.retry(job["job_id"])
        await self.scheduler.process(job["job_id"])
        self.assertEqual(self.scheduler.get(job["job_id"])["status"], "partial")

    async def test_cancel_requeues_with_saved_checkpoint(self):
        job = self.job_with_plan(2)
        self.provider.side_effect = [{"translation_text": "First complete page"}, asyncio.CancelledError()]
        with self.assertRaises(asyncio.CancelledError):
            await self.scheduler.process(job["job_id"])
        self.assertEqual(self.scheduler.get(job["job_id"])["status"], "queued")
        self.assertEqual(self.scheduler.artifact(job["job_id"])["coverage"]["done"], 1)

    async def test_provider_outage_stops_after_two_calls_and_retains_pending_pages(self):
        job = self.job_with_plan(3)
        self.provider.side_effect = module.TranslationError("Temporarily unreachable", retryable=True, stop_job=True)
        with patch.object(module.asyncio, "sleep", new=AsyncMock()):
            await self.scheduler.process(job["job_id"])
        self.assertEqual(self.provider.await_count, 2)
        self.assertEqual(self.scheduler.artifact(job["job_id"])["coverage"], {"total": 3, "done": 0, "failed": 1, "pending": 2})

    async def test_mismatched_checkpoint_is_not_reused(self):
        job = self.job_with_plan(1)
        save(self.scheduler.directory(job["job_id"]) / "unit0.json", {"status": "done", "original_sha256": "wrong", "translation_text": "Wrong source"})
        self.assertEqual(self.scheduler.artifact(job["job_id"])["coverage"]["failed"], 1)
        await self.scheduler.process(job["job_id"])
        self.provider.assert_awaited_once()

    async def test_changed_implementation_requires_new_job(self):
        job = self.job_with_plan(1)
        self.scheduler.update(job["job_id"], implementation_sha256="old")
        await self.scheduler.process(job["job_id"])
        self.assertEqual(self.scheduler.get(job["job_id"])["status"], "superseded")
        self.provider.assert_not_awaited()


class TranslationInputTests(TranslationFixture, TestCase):
    def setUp(self):
        self.fixture()

    def test_splitting_covers_every_original_character_once(self):
        text = ("中文 spaces \n" * 1000) + "尾"
        parts = list(module.split_text(text))
        self.assertEqual("".join(piece for _, _, piece in parts), text)
        self.assertTrue(all(text[start:end] == piece and len(piece) <= module.MAX_INPUT_CHARACTERS for start, end, piece in parts))
        self.assertEqual(parts[0][0], 0)
        self.assertEqual(parts[-1][1], len(text))

    def test_foreign_file_is_rejected_and_model_changes_job_identity(self):
        with self.assertRaises(ValueError):
            self.scheduler.enqueue("source", "another-paper")
        first = self.scheduler.enqueue("source")
        with patch.object(module, "configured_model", return_value="different-model"):
            self.assertNotEqual(self.scheduler.enqueue("source")["job_id"], first["job_id"])

    def test_plan_records_real_representation_and_chunk_offsets(self):
        job = self.scheduler.enqueue("source")
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.execute.return_value = [{"id": "paper", "stored_path": "/fixture/paper.pdf", "original_filename": "中文.pdf"}]
        representation = {"source_sha256": "a"*64, "representation_id": "representation-hash", "status": "text_ready", "warnings": [],
                          "units": [{"text": "中文" * 2001, "method": "ocr", "locator": "page:1", "page_index": 1, "confidence": .4}]}
        with patch("app.api.repository.get_connection", return_value=connection), patch("app.api.repository.ensure_material"), \
             patch("app.repository.document_representation.extract_document", return_value=representation):
            plan = self.scheduler.plan(job)
        self.assertEqual("".join(unit["original_text"] for unit in plan["units"]), representation["units"][0]["text"])
        self.assertEqual(plan["units"][0]["provenance"]["representation_id"], "representation-hash")
        self.assertEqual(plan["units"][0]["source_confidence"], .4)


class TranslationProviderTests(IsolatedAsyncioTestCase):
    async def request(self, response):
        sent = []
        async def handler(request):
            sent.append(json.loads(request.content))
            return response
        client_class = httpx.AsyncClient
        def client(**kwargs):
            return client_class(transport=httpx.MockTransport(handler), **kwargs)
        with patch("app.api.repository.ai_chat_configured", return_value=True), \
             patch("app.api.repository.ollama_endpoint", return_value="https://provider.invalid/api/chat"), \
             patch("app.api.repository.ollama_headers", return_value={}), patch.object(module.httpx, "AsyncClient", side_effect=client):
            result = await module.translate_text("原文", "gpt-oss:120b")
        return result, sent

    async def test_success_has_bounded_budget_and_safe_metadata(self):
        result, sent = await self.request(httpx.Response(200, json={"done": True, "done_reason": "stop", "eval_count": 10,
                      "message": {"content": "Complete English translation", "thinking": "private reasoning"}}))
        self.assertEqual(result["translation_text"], "Complete English translation")
        self.assertNotIn("private reasoning", json.dumps(result))
        self.assertEqual(sent[0]["think"], "low")
        self.assertEqual(sent[0]["options"]["num_predict"], 4096)

    async def test_missing_provider_is_permanent_and_makes_no_request(self):
        with patch("app.api.repository.ai_chat_configured", return_value=False), patch.object(module.httpx, "AsyncClient") as client:
            with self.assertRaises(module.TranslationError) as failure:
                await module.translate_text("原文", "gpt-oss:120b")
        self.assertFalse(failure.exception.retryable)
        client.assert_not_called()

    async def test_http_error_does_not_expose_upstream_body(self):
        for status in (401, 404, 429, 503):
            with self.subTest(status=status):
                with self.assertRaises(module.TranslationError) as failure:
                    await self.request(httpx.Response(status, text="secret provider response"))
                self.assertEqual(failure.exception.retryable, status in {429, 503})
                self.assertNotIn("secret", str(failure.exception))

    async def test_incomplete_invalid_and_oversized_outputs_are_rejected(self):
        responses = [httpx.Response(200, json={"done": True, "done_reason": "length", "message": {"content": "Partial English"}}),
                     httpx.Response(200, text="not json"),
                     httpx.Response(200, json={"done": True, "message": {"content": "中文"}}),
                     httpx.Response(200, content=b"x" * (module.MAX_RESPONSE_BYTES+1))]
        for response in responses:
            with self.assertRaises(module.TranslationError):
                await self.request(response)
