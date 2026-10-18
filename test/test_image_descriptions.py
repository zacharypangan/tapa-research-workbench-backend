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

    def json(self):
        return {"message": {"content": self.raw}}


class ImageDescriptionTests(unittest.TestCase):
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
                self.assertEqual(asyncio.run(repository.request_image_caption(str(image), "context")), "map; fiji; scale bar")
                record = image_descriptions.read_record(temp, inputs["key"])
                self.assertEqual(record["status"], "complete")
                self.assertEqual(len(record["attempts"]), 2)


if __name__ == "__main__":
    unittest.main()
