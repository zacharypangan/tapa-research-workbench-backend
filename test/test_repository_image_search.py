"""Image indexing/search use the same configured embedding model, with no network."""

import asyncio
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from app.api import repository
from app.repository.schemas import MultimodalSearchRequest


class RepositoryImageSearchTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for symbol, value in {"STORAGE_ROOT": self.root, "FILES_ROOT": self.root / "files",
                              "IMAGES_ROOT": self.root / "images", "DB_PATH": self.root / "repository.sqlite"}.items():
            self.stack.enter_context(patch.object(repository, symbol, str(value)))
        repository.init_repository_db()
        self.stack.enter_context(patch.object(repository, "ai_configured", return_value=True))
        self.stack.enter_context(patch.object(repository, "is_informative_image", return_value=True))
        self.stack.enter_context(patch.object(repository, "request_image_caption", new=AsyncMock(return_value="A wooden beater.")))
        self.image_path = self.root / "synthetic.png"
        self.image_path.write_bytes(b"synthetic image fixture")
        self.embedding = self.stack.enter_context(patch.object(
            repository, "request_embedding", new=AsyncMock(return_value=[1.0, 0.0])))
        with repository.get_connection() as con:
            for material_id in ("source", "other"):
                con.execute("""INSERT INTO materials
                    (id, title, source_type, status, created_at, updated_at)
                    VALUES (?, ?, 'other', 'uploaded', 'test', 'test')""",
                            (material_id, material_id))
                con.execute("""INSERT INTO image_evidence
                    (id, material_id, evidence_type, source_kind, source_locator,
                     page_ref, page_index, image_path, extraction_method, ocr_text,
                     visual_caption, created_at)
                    VALUES (?, ?, 'document_image', 'file_pdf', 'synthetic.pdf',
                            'page:1', 1, ?, 'test', 'wooden beater',
                            'A wooden beater.', 'test')""", (material_id + "-image", material_id, str(self.image_path)))

    def check_provider_roundtrip(self, provider, model):
        with patch.object(repository, "EMBEDDING_PROVIDER", provider), \
                patch.object(repository, "GEMINI_EMBEDDING_MODEL", "synthetic-gemini"), \
                patch.object(repository, "GEMINI_EMBEDDING_DIMENSIONS", 768), \
                patch.object(repository, "OLLAMA_EMBEDDING_MODEL", "synthetic-ollama"):
            with repository.get_connection() as con:
                indexed = asyncio.run(repository.index_missing_image_embeddings(con))
                self.assertEqual(indexed["indexed_count"], 2)
                self.assertEqual(indexed["model"], model)
            result = asyncio.run(repository.search_image_evidence(
                MultimodalSearchRequest(query="wooden beater", material_id="source")))
            self.assertEqual([row["image_id"] for row in result["image_results"]], ["source-image"])
            self.assertEqual(result["image_results"][0]["semantic_score"], 1.0)
            self.embedding.assert_awaited_with("wooden beater", task_type="RETRIEVAL_QUERY")
            with repository.get_connection() as con:
                con.execute("UPDATE image_embeddings SET method_version = 'outdated' WHERE material_id = 'source'")
            result = asyncio.run(repository.search_image_evidence(
                MultimodalSearchRequest(query="wooden beater", material_id="source")))
            self.assertEqual(result["image_results"], [])

    def test_gemini_indexed_images_are_searchable(self):
        self.check_provider_roundtrip("gemini", "gemini:synthetic-gemini:768")

    def test_ollama_indexed_images_remain_searchable(self):
        self.check_provider_roundtrip("ollama", "synthetic-ollama")


if __name__ == "__main__":
    unittest.main()
