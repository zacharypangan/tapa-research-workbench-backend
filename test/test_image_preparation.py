"""Independent image preparation: outages, reuse and concurrent changes."""

import asyncio
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from app.api import repository as repo
from app.repository import image_descriptions
from app.repository.image_preparation import prepare_images


class ImagePreparationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for symbol, value in {"STORAGE_ROOT": self.root, "FILES_ROOT": self.root / "files",
                              "IMAGES_ROOT": self.root / "images", "DB_PATH": self.root / "repo.sqlite"}.items():
            self.stack.enter_context(patch.object(repo, symbol, str(value)))
        repo.init_repository_db()
        self.image = self.root / "image.png"
        self.image.write_bytes(b"fixture image")
        self.stack.enter_context(patch.object(repo, "is_informative_image", return_value=True))
        self.ocr = self.stack.enter_context(patch("app.repository.image_preparation._read_ocr", return_value={"text": "石拍", "state": "readable"}))
        self.stack.enter_context(patch.object(repo, "get_image_context_text", return_value="source context"))
        self.embedding = self.stack.enter_context(patch.object(repo, "request_embedding", new=AsyncMock(side_effect=AssertionError("No vectors required"))))
        self.stack.enter_context(patch.object(repo, "ai_embedding_configured", return_value=False))
        self.caption = self.stack.enter_context(patch.object(repo, "request_image_caption", new=AsyncMock(return_value="stone; tool")))
        with repo.get_connection() as con:
            con.execute("""INSERT INTO materials (id,title,source_type,status,language,created_at,updated_at)
                VALUES ('fixture','Paper','other','uploaded','zh','test','test')""")
            con.execute("""INSERT INTO image_evidence (id,material_id,evidence_type,source_kind,source_locator,
                page_ref,page_index,image_path,extraction_method,ocr_text,visual_caption,created_at)
                VALUES ('image','fixture','document_image','file_pdf','paper.pdf','page:1',1,?,'test','','','test')""", (str(self.image),))

    def run_batch(self, **kwargs):
        with repo.get_connection() as con:
            return asyncio.run(prepare_images(con, "fixture", **kwargs))

    def test_descriptions_and_ocr_work_without_embeddings(self):
        result = self.run_batch()
        self.assertEqual(result["captioned_count"], 1)
        self.assertEqual(result["ocr_count"], 1)
        self.assertFalse(result["embedding_required"])
        self.embedding.assert_not_awaited()
        with repo.get_connection() as con:
            image = con.execute("SELECT ocr_text,visual_caption FROM image_evidence").fetchone()
            self.assertEqual(tuple(image), ("石拍", "stone; tool"))
            self.assertEqual(con.execute("SELECT COUNT(*) FROM image_embeddings").fetchone()[0], 0)

    def test_failed_vision_preserves_ocr_and_previous_caption(self):
        self.caption.return_value = ""
        with repo.get_connection() as con:
            con.execute("UPDATE image_evidence SET visual_caption='previous labels'")
        with patch.object(repo, "_caption_failure", return_value={"error_category": "rate_limit", "retryable": True}):
            result = self.run_batch()
        self.assertEqual(result["failed_count"], 1)
        self.assertTrue(result["outcomes"][0]["ocr_searchable"])
        with repo.get_connection() as con:
            self.assertEqual(tuple(con.execute("SELECT ocr_text,visual_caption FROM image_evidence").fetchone()), ("石拍", "previous labels"))

    def test_current_cached_description_avoids_provider_and_skip_ids_bound_pass(self):
        inputs = image_descriptions.description_input(str(self.image), "source context", repo.OLLAMA_VISION_MODEL)
        image_descriptions.save_record(repo.STORAGE_ROOT, {**inputs, "status": "complete", "normalized_caption": "stone; tool"})
        with repo.get_connection() as con:
            con.execute("UPDATE image_evidence SET visual_caption='stone; tool',ocr_text='石拍'")
        self.assertEqual(self.run_batch()["processed_image_count"], 0)
        self.caption.assert_not_awaited()
        with repo.get_connection() as con:
            con.execute("UPDATE image_evidence SET visual_caption=''")
        self.assertEqual(self.run_batch(skip_ids={"image"})["pending_count"], 0)

    def test_changed_source_bytes_do_not_receive_stale_caption(self):
        async def change(*args):
            self.image.write_bytes(b"replacement image with different bytes")
            return "outdated labels"
        self.caption.side_effect = change
        result = self.run_batch()
        self.assertEqual(result["outcomes"][0]["error_category"], "source_changed")
        with repo.get_connection() as con:
            self.assertEqual(con.execute("SELECT visual_caption FROM image_evidence").fetchone()[0], "")

    def test_missing_source_is_visible_and_never_sent_to_provider(self):
        self.image.unlink()
        result = self.run_batch()
        self.assertEqual(result["outcomes"][0]["error_category"], "source_unavailable")
        self.caption.assert_not_awaited()

    def test_missing_ocr_runtime_keeps_successful_caption_and_remains_retryable_after_configuration_fix(self):
        self.ocr.return_value = {"text": "", "state": "unavailable", "error_category": "ocr_not_available", "retryable": False}
        result = self.run_batch()
        self.assertEqual(result["captioned_count"], 1)
        self.assertEqual(result["outcomes"][0]["error_category"], "ocr_not_available")
        with repo.get_connection() as con:
            self.assertEqual(con.execute("SELECT visual_caption FROM image_evidence").fetchone()[0], "stone; tool")

    def test_invalid_limits_rejected(self):
        for value in (0, 201):
            with self.assertRaises(ValueError):
                self.run_batch(limit=value)
