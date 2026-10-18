"""Description caching and raw/failure provenance, using mocked HTTP only."""
import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from app.api import repository
from app.repository import image_descriptions


class Client:
    def __init__(self, status=200, raw="Map; Fiji; scale bar"):
        self.post = AsyncMock(return_value=self)
        self.status_code = status
        self.raw = raw

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def stream(self, method, url, **kwargs):
        client = self
        class ResponseContext:
            async def __aenter__(self):
                return await client.post(url, **kwargs)
            async def __aexit__(self, *args):
                pass
        return ResponseContext()

    async def aiter_bytes(self, chunk_size=8192):
        import json
        payload = json.dumps(self.json()).encode()
        for offset in range(0, len(payload), chunk_size):
            yield payload[offset:offset + chunk_size]

    def json(self):
        return {"message": {"content": self.raw}}


class ImageDescriptionTests(unittest.TestCase):
    def setUp(self):
        repository.clear_image_description_failure_guard()
        self.addCleanup(repository.clear_image_description_failure_guard)

    def test_context_model_pixel_cache_keys_raw_provenance_and_inflight_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "image.png"
            image.write_bytes(b"synthetic pixels")
            client = Client()
            with patch.object(repository, "STORAGE_ROOT", temp), patch.object(repository, "OLLAMA_VISION_MODEL", "test-vision"), patch.object(repository.httpx, "AsyncClient", return_value=client):
                async def first():
                    values = await asyncio.gather(repository.request_image_caption(str(image), "context"), repository.request_image_caption(str(image), "context"))
                    self.assertEqual(values, ["map; fiji; scale bar"] * 2)
                    self.assertEqual(await repository.request_image_caption(str(image), "context"), values[0])
                asyncio.run(first())
                self.assertEqual(client.post.await_count, 1)
                inputs = image_descriptions.description_input(str(image), "context", "test-vision")
                record = image_descriptions.read_record(temp, inputs["key"])
                self.assertEqual(record["raw_response"], "Map; Fiji; scale bar")
                self.assertEqual(record["normalized_caption"], "map; fiji; scale bar")
                self.assertEqual(record["evidence_origin"], "machine_visual_description")
                asyncio.run(repository.request_image_caption(str(image), "different context"))
                self.assertEqual(client.post.await_count, 2)
                with patch.object(repository, "OLLAMA_VISION_MODEL", "new-model"):
                    asyncio.run(repository.request_image_caption(str(image), "different context"))
                self.assertEqual(client.post.await_count, 3)
                image.write_bytes(b"changed image pixels")
                asyncio.run(repository.request_image_caption(str(image), "context"))
                self.assertEqual(client.post.await_count, 4)

    def test_failure_categories_and_retry_preserve_attempt_history(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "image.png"
            image.write_bytes(b"synthetic pixels")
            client = Client(status=401)
            with patch.object(repository, "STORAGE_ROOT", temp), patch.object(repository, "OLLAMA_VISION_MODEL", "test-vision"), patch.object(repository.httpx, "AsyncClient", return_value=client):
                self.assertEqual(asyncio.run(repository.request_image_caption(str(image), "context")), "")
                self.assertEqual(client.post.await_count, 1)
                inputs = image_descriptions.description_input(str(image), "context", "test-vision")
                record = image_descriptions.read_record(temp, inputs["key"])
                self.assertEqual(record["failure_category"], "authentication")
                self.assertEqual(record["raw_response"], "")
                client.status_code = 200
                repository.clear_image_description_failure_guard()
                self.assertEqual(asyncio.run(repository.request_image_caption(str(image), "context")), "map; fiji; scale bar")
                record = image_descriptions.read_record(temp, inputs["key"])
                self.assertEqual(record["status"], "complete")
                self.assertEqual(len(record["attempts"]), 2)

    def test_permanent_404_is_one_request_and_guards_other_images_until_explicit_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "one.png", Path(temp) / "two.png"
            first.write_bytes(b"first image")
            second.write_bytes(b"second image")
            client = Client(status=404)
            with patch.object(repository, "STORAGE_ROOT", temp), patch.object(repository, "OLLAMA_VISION_MODEL", "missing-fixture-model"), patch.object(repository.httpx, "AsyncClient", return_value=client):
                self.assertEqual(asyncio.run(repository.request_image_caption(str(first), "context")), "")
                self.assertEqual(client.post.await_count, 1)
                diagnostics = repository.image_description_diagnostics()
                self.assertEqual(diagnostics["provider_failure"]["http_status"], 404)
                self.assertFalse(diagnostics["provider_failure"]["retryable"])
                self.assertLessEqual(diagnostics["provider_failure"]["cooldown_seconds_remaining"], 300)
                self.assertEqual(asyncio.run(repository.request_image_caption(str(second), "context")), "")
                self.assertEqual(client.post.await_count, 1)
                inputs = image_descriptions.description_input(str(second), "context", "missing-fixture-model")
                record = image_descriptions.read_record(temp, inputs["key"])
                self.assertTrue(record["provider_request_skipped"])
                self.assertEqual(record["attempts"], [])
                failure = repository._caption_failure(inputs)
                self.assertEqual(failure["http_status"], 404)
                self.assertFalse(failure["retryable"])
                # An explicit user retry may immediately probe a corrected provider.
                repository.clear_image_description_failure_guard()
                client.status_code = 200
                self.assertEqual(asyncio.run(repository.request_image_caption(str(second), "context")), "map; fiji; scale bar")
                self.assertEqual(client.post.await_count, 2)
                self.assertIsNone(repository.image_description_diagnostics()["provider_failure"])

    def test_other_permanent_4xx_are_not_retried_internally(self):
        for status in (400, 405, 413, 422):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                image = Path(temp) / "image.png"
                image.write_bytes(b"synthetic pixels")
                client = Client(status=status)
                with patch.object(repository, "STORAGE_ROOT", temp), patch.object(repository, "OLLAMA_VISION_MODEL", "fixture"), patch.object(repository.httpx, "AsyncClient", return_value=client):
                    self.assertEqual(asyncio.run(repository.request_image_caption(str(image), "context")), "")
                    self.assertEqual(client.post.await_count, 1)
                    inputs = image_descriptions.description_input(str(image), "context", "fixture")
                    self.assertFalse(repository._caption_failure(inputs)["retryable"])
                    # A rejected image request does not prove every image is invalid.
                    self.assertIsNone(repository.image_description_diagnostics()["provider_failure"])

    def test_transient_failure_has_two_attempt_limit_and_generation_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "image.png"
            image.write_bytes(b"synthetic pixels")
            client = Client(status=503)
            with patch.object(repository, "STORAGE_ROOT", temp), patch.object(repository, "OLLAMA_VISION_MODEL", "fixture"), patch.object(repository.httpx, "AsyncClient", return_value=client), patch.object(repository.asyncio, "sleep", new=AsyncMock()):
                self.assertEqual(asyncio.run(repository.request_image_caption(str(image), "context")), "")
                self.assertEqual(client.post.await_count, 2)
                self.assertEqual(client.post.await_args.kwargs["json"]["options"]["num_predict"], 256)
                inputs = image_descriptions.description_input(str(image), "context", "fixture")
                self.assertTrue(repository._caption_failure(inputs)["retryable"])

    def test_oversized_output_is_not_accepted_or_retried_and_provenance_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "image.png"
            image.write_bytes(b"synthetic pixels")
            raw = "x" * (repository._IMAGE_DESCRIPTION_RAW_LIMIT + 100)
            client = Client(raw=raw)
            with patch.object(repository, "STORAGE_ROOT", temp), patch.object(repository, "OLLAMA_VISION_MODEL", "fixture"), patch.object(repository.httpx, "AsyncClient", return_value=client):
                self.assertEqual(asyncio.run(repository.request_image_caption(str(image), "context")), "")
                self.assertEqual(client.post.await_count, 1)
                inputs = image_descriptions.description_input(str(image), "context", "fixture")
                record = image_descriptions.read_record(temp, inputs["key"])
                self.assertEqual(record["failure_category"], "response_limit_exceeded")
                self.assertTrue(record["raw_response_truncated"])
                self.assertEqual(len(record["raw_response"]), repository._IMAGE_DESCRIPTION_RAW_LIMIT)
                self.assertEqual(record["raw_response_char_count"], len(raw))
                self.assertFalse(repository._caption_failure(inputs)["retryable"])
                self.assertEqual(record["normalized_caption"], "")

    def test_oversized_http_response_is_stopped_before_full_body_is_read(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "image.png"
            image.write_bytes(b"synthetic pixels")
            client = Client(raw="x" * (repository._IMAGE_DESCRIPTION_RESPONSE_LIMIT * 4))
            reads = []
            async def chunks(chunk_size=8192):
                for index in range(32):
                    reads.append(index)
                    yield b"x" * chunk_size
            client.aiter_bytes = chunks
            with patch.object(repository, "STORAGE_ROOT", temp), patch.object(repository, "OLLAMA_VISION_MODEL", "fixture"), patch.object(repository.httpx, "AsyncClient", return_value=client):
                self.assertEqual(asyncio.run(repository.request_image_caption(str(image), "context")), "")
                self.assertEqual(client.post.await_count, 1)
                self.assertEqual(len(reads), repository._IMAGE_DESCRIPTION_RESPONSE_LIMIT // 8192 + 1)
                inputs = image_descriptions.description_input(str(image), "context", "fixture")
                record = image_descriptions.read_record(temp, inputs["key"])
                self.assertEqual(record["failure_category"], "response_limit_exceeded")
                self.assertEqual(record["raw_response"], "")

    def test_input_and_context_budgets_skip_provider_with_safe_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "image.png"
            image.write_bytes(b"synthetic pixels")
            client = Client()
            with patch.object(repository, "STORAGE_ROOT", temp), patch.object(repository, "OLLAMA_VISION_MODEL", "fixture"), patch.object(repository.httpx, "AsyncClient", return_value=client):
                with patch.object(repository, "_IMAGE_DESCRIPTION_INPUT_LIMIT", 8):
                    self.assertEqual(asyncio.run(repository.request_image_caption(str(image), "context")), "")
                inputs = image_descriptions.description_input(str(image), "context", "fixture")
                self.assertEqual(repository._caption_failure(inputs)["error_category"], "input_limit_exceeded")
                long_context = "x" * (repository._IMAGE_DESCRIPTION_CONTEXT_LIMIT + 1)
                self.assertEqual(asyncio.run(repository.request_image_caption(str(image), long_context)), "")
                inputs = image_descriptions.description_input(str(image), long_context, "fixture")
                self.assertEqual(repository._caption_failure(inputs)["error_category"], "context_limit_exceeded")
                client.post.assert_not_awaited()

    def test_success_cache_remains_usable_during_model_guard_and_guard_expires(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "one.png", Path(temp) / "two.png"
            first.write_bytes(b"first image")
            second.write_bytes(b"second image")
            client = Client()
            with patch.object(repository, "STORAGE_ROOT", temp), patch.object(repository, "OLLAMA_VISION_MODEL", "fixture"), patch.object(repository.httpx, "AsyncClient", return_value=client):
                self.assertEqual(asyncio.run(repository.request_image_caption(str(first), "context")), "map; fiji; scale bar")
                client.status_code = 404
                self.assertEqual(asyncio.run(repository.request_image_caption(str(second), "context")), "")
                self.assertEqual(client.post.await_count, 2)
                self.assertEqual(asyncio.run(repository.request_image_caption(str(first), "context")), "map; fiji; scale bar")
                self.assertEqual(client.post.await_count, 2)
                repository._IMAGE_DESCRIPTION_FAILURE_GUARD[repository._image_description_guard_key()]["expires_at"] = 0
                client.status_code = 200
                self.assertEqual(asyncio.run(repository.request_image_caption(str(second), "context")), "map; fiji; scale bar")
                self.assertEqual(client.post.await_count, 3)


if __name__ == "__main__":
    unittest.main()
