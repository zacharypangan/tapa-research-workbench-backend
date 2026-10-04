"""Search transparency regressions using isolated repositories and synthetic evidence."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.api import repository
from app.repository.schemas import AskCorpusRequest, MultimodalSearchRequest, SemanticSearchRequest
from test import test_production_fixes as fixtures


class RepositoryTransparencyTests(unittest.TestCase):
    setUp = fixtures.ProductionFixTests.setUp
    tearDown = fixtures.ProductionFixTests.tearDown
    material = fixtures.ProductionFixTests.material
    extract = fixtures.ProductionFixTests.extract
    segment = fixtures.ProductionFixTests.segment

    def test_empty_coverage_is_available_without_provider_calls(self):
        with patch.object(repository, "check_ollama_generation_readiness", AsyncMock(side_effect=AssertionError("No provider check"))), patch.object(repository, "check_embedding_readiness", AsyncMock(side_effect=AssertionError("No provider check"))):
            response = self.client.get("/api/v1/repository/ai/coverage")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["scope"], "repository")
        self.assertIsNone(result["material_id"])
        for field in ("material_count", "uploaded_file_count", "segment_count", "embedded_segment_count", "pending_segment_count", "image_evidence_count", "embedded_image_count", "pending_image_count", "materials_without_text_count", "materials_with_failed_extraction_count"):
            self.assertEqual(result[field], 0, field)

    def test_coverage_is_scoped_and_counts_only_current_indexes_and_latest_failures(self):
        selected = self.material(upload=True)
        other = self.material(upload=True)
        self.material()  # A metadata-only reference has no extracted text.
        self.extract(selected)
        self.extract(other)
        selected_segment = self.segment(selected)
        other_segment = self.segment(other)
        with repository.get_connection() as con:
            con.execute("INSERT INTO segment_embeddings VALUES (?, ?, ?, '[1,0]', '2026-10-05')", (selected_segment["id"], selected, repository.active_embedding_model_name()))
            con.execute("INSERT INTO segment_embeddings VALUES (?, ?, 'obsolete-model', '[1,0]', '2026-10-05')", (other_segment["id"], other))
            for image_id, material_id, version in (("current", selected, repository.MULTIMODAL_METHOD_VERSION), ("outdated", selected, "obsolete-method"), ("other", other, repository.MULTIMODAL_METHOD_VERSION)):
                con.execute("INSERT INTO image_evidence(id, material_id, evidence_type, source_kind, source_locator, page_ref, page_index, image_path, extraction_method, created_at) VALUES (?, ?, 'image', 'file_image', 'source.png', '1', 1, 'synthetic.png', 'synthetic', '2026-10-05')", (image_id, material_id))
                con.execute("INSERT INTO image_embeddings VALUES (?, ?, ?, ?, '[1,0]', '2026-10-05')", (image_id, material_id, repository.active_embedding_model_name(), version))
            for run_id, material_id, status, created_at in (("old-failure", selected, "failed", "2027-01-01"), ("new-success", selected, "completed", "2027-01-02"), ("current-failure", other, "error", "2027-01-02")):
                con.execute("INSERT INTO extraction_runs VALUES (?, ?, 0, 0, 0, 0, 0, ?, 'synthetic', ?)", (run_id, material_id, status, created_at))
            con.execute("INSERT INTO files(id, material_id, original_filename, stored_path, file_size, uploaded_at, parser_status) VALUES ('failed-file', ?, 'append.txt', 'synthetic.txt', 10, '2026-10-05', 'failed')", (selected,))
            con.execute("INSERT INTO files(id, material_id, original_filename, stored_path, file_size, uploaded_at, parser_status) VALUES ('unsupported-file', ?, 'append.bin', 'synthetic.bin', 10, '2026-10-05', 'unsupported')", (other,))
        global_result = self.client.get("/api/v1/repository/ai/coverage").json()
        result = self.client.get("/api/v1/repository/ai/coverage", params={"material_id": selected}).json()
        self.assertEqual(global_result["material_count"], 3)
        self.assertEqual(global_result["materials_without_text_count"], 1)
        self.assertEqual(global_result["materials_with_failed_extraction_count"], 1)
        self.assertEqual(global_result["uploaded_file_count"], 4)
        self.assertEqual(global_result["files_without_extracted_evidence_count"], 2)
        self.assertEqual(global_result["files_with_failed_extraction_count"], 1)
        self.assertEqual(global_result["files_with_unsupported_format_count"], 1)
        self.assertEqual(global_result["segment_count"], 2)
        self.assertEqual(global_result["embedded_segment_count"], 1)
        self.assertEqual(global_result["pending_segment_count"], 1)
        self.assertEqual(global_result["image_evidence_count"], 3)
        self.assertEqual(global_result["embedded_image_count"], 2)
        self.assertEqual(global_result["pending_image_count"], 1)
        self.assertEqual(result["scope"], "material")
        self.assertEqual(result["material_id"], selected)
        self.assertEqual(result["material_count"], 1)
        self.assertEqual(result["materials_without_text_count"], 0)
        self.assertEqual(result["materials_with_failed_extraction_count"], 0)
        self.assertEqual(result["uploaded_file_count"], 2)
        self.assertEqual(result["segment_count"], 1)
        self.assertEqual(result["pending_segment_count"], 0)
        self.assertEqual(result["image_evidence_count"], 2)
        self.assertEqual(result["embedded_image_count"], 1)
        self.assertEqual(result["pending_image_count"], 1)

    def test_coverage_validates_material_filter(self):
        self.assertEqual(self.client.get("/api/v1/repository/ai/coverage", params={"material_id": "missing"}).status_code, 404)
        self.assertEqual(self.client.get("/api/v1/repository/ai/coverage", params={"material_id": ""}).status_code, 422)

    def test_coverage_initializes_an_empty_repository(self):
        with patch.object(repository, "DB_PATH", str(self.root / "empty-repository.sqlite")):
            response = self.client.get("/api/v1/repository/ai/coverage")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["material_count"], 0)

    def test_evidence_report_keeps_exact_retrieval_provenance(self):
        material_id = self.material(upload=True)
        self.extract(material_id)
        with patch.object(repository, "ai_embedding_configured", return_value=False):
            response = self.client.post("/api/v1/repository/ai/evidence-report", json={"query": "Synthetic", "material_id": material_id})
        self.assertEqual(response.status_code, 200, response.text)
        passage = response.json()["themes"][0]["passages"][0]
        self.assertEqual(passage["retrieval_method"], "exact_text")
        self.assertIsNone(passage["semantic_score"])

    def test_retrieval_methods_distinguish_embedding_and_exact_fallback(self):
        material_id = self.material(upload=True)
        self.extract(material_id)
        segment = self.segment(material_id)
        with repository.get_connection() as con:
            con.execute("INSERT INTO segment_embeddings VALUES (?, ?, ?, '[1,0]', '2026-10-05')", (segment["id"], material_id, repository.active_embedding_model_name()))
            con.execute("INSERT INTO extracted_segments(material_id, source_kind, source_locator, page_ref, page_index, content_text, char_count, created_at) VALUES (?, 'file', 'append.txt', '2', 2, 'Synthetic unindexed evidence', 27, '2026-10-05')", (material_id,))
            con.execute("INSERT INTO image_evidence(id, material_id, evidence_type, source_kind, source_locator, page_ref, page_index, image_path, extraction_method, ocr_text, created_at) VALUES ('image', ?, 'image', 'file_image', 'source.png', '1', 1, 'synthetic.png', 'synthetic', 'Synthetic image evidence', '2026-10-05')", (material_id,))
        with patch.object(repository, "ai_embedding_configured", return_value=True), patch.object(repository, "request_embedding", AsyncMock(return_value=[1.0, 0.0])):
            result = asyncio.run(repository.semantic_search_segments(SemanticSearchRequest(query="Synthetic", material_id=material_id)))
        results = {item["source_locator"]: item for item in result["results"]}
        self.assertEqual(results["source.txt"]["retrieval_method"], "embedding_similarity")
        self.assertEqual(results["source.txt"]["semantic_score"], 1.0)
        self.assertEqual(results["append.txt"]["retrieval_method"], "exact_text")
        self.assertIsNone(results["append.txt"]["semantic_score"])
        self.assertEqual(results["append.txt"]["matched_terms"], ["Synthetic"])
        with patch.object(repository, "ai_embedding_configured", return_value=False):
            offline = asyncio.run(repository.semantic_search_segments(SemanticSearchRequest(query="Synthetic", material_id=material_id)))
            images = asyncio.run(repository.search_image_evidence(MultimodalSearchRequest(query="Synthetic", material_id=material_id)))
        self.assertTrue(all(item["retrieval_method"] == "exact_text" and item["semantic_score"] is None for item in offline["results"]))
        self.assertEqual(images["image_results"][0]["retrieval_method"], "exact_text")
        self.assertIsNone(images["image_results"][0]["semantic_score"])

    def test_generated_answer_context_matches_provider_excerpts_and_excludes_images(self):
        evidence = [{"segment_id": 7, "material_id": "material", "material_title": "Synthetic source", "page_ref": "12", "source_locator": "source.txt", "content_text": "x" * 1600 + "excluded tail", "citation": "Synthetic citation"}]
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.return_value = httpx.Response(200, json={"message": {"content": "Draft answer [1]."}})
        with patch.object(repository, "ollama_available", AsyncMock(return_value=True)), patch.object(repository, "semantic_search_segments", AsyncMock(return_value={"results": evidence, "related_observations": []})), patch.object(repository, "search_image_evidence", AsyncMock(return_value={"image_results": [{"image_id": "displayed-only"}]})), patch.object(repository.httpx, "AsyncClient", return_value=client):
            result = asyncio.run(repository.ask_corpus(AskCorpusRequest(question="What is supported?")))
        context = result["generation_context"]
        self.assertFalse(context["images_included"])
        self.assertEqual(context["passage_character_limit"], 1600)
        passage = context["passages"][0]
        self.assertEqual(passage["citation_number"], 1)
        self.assertEqual(passage["segment_id"], 7)
        self.assertEqual(passage["excerpt"], "x" * 1600)
        self.assertTrue(passage["truncated"])
        user_input = client.post.call_args.kwargs["json"]["messages"][1]["content"]
        self.assertIn(f"[1] Synthetic source (12): {passage['excerpt']}", user_input)
        self.assertNotIn("excluded tail", user_input)
        self.assertNotIn("displayed-only", user_input)

    def test_image_matches_identify_ocr_ai_description_researcher_labels_and_metadata(self):
        material_id = self.material()
        with repository.get_connection() as con:
            for image_id, locator, ocr, caption in (
                ("ocr", "source.png", "Orchard inscription", "Orchard scene"),
                ("caption", "source.png", "Unreadable", "Orchard scene"),
                ("labels", "source.png", "Unreadable", "Blue shapes"),
                ("metadata", "Orchard.png", "Unreadable", "Blue shapes"),
            ):
                con.execute("INSERT INTO image_evidence(id, material_id, evidence_type, source_kind, source_locator, page_ref, page_index, image_path, extraction_method, ocr_text, visual_caption, created_at) VALUES (?, ?, 'image', 'file_image', ?, '1', 1, 'synthetic.png', 'synthetic', ?, ?, '2026-10-05')", (image_id, material_id, locator, ocr, caption))
            con.execute("INSERT INTO observations(id, material_id, source_image_id, observation_type, observed_text, created_at, updated_at) VALUES ('label', ?, 'labels', 'note', 'Orchard researcher label', '2026-10-05', '2026-10-05')", (material_id,))
        with patch.object(repository, "ai_embedding_configured", return_value=False):
            result = asyncio.run(repository.search_image_evidence(MultimodalSearchRequest(query="Orchard", material_id=material_id)))
        images = {image["image_id"]: image for image in result["image_results"]}
        self.assertEqual(images["ocr"]["matched_text_sources"], ["ocr_text", "visual_caption"])
        self.assertEqual(images["ocr"]["evidence_level"], "direct_image_text_match")
        self.assertEqual(images["caption"]["matched_text_sources"], ["visual_caption"])
        self.assertEqual(images["caption"]["evidence_level"], "ai_description_text_match")
        self.assertIn("AI image description", images["caption"]["retrieval_basis"])
        self.assertNotIn("OCR", images["caption"]["retrieval_basis"])
        self.assertEqual(images["labels"]["matched_text_sources"], ["observation_labels"])
        self.assertEqual(images["labels"]["evidence_level"], "observation_label_match")
        self.assertIn("researcher observation labels", images["labels"]["retrieval_basis"])
        self.assertEqual(images["metadata"]["matched_text_sources"], ["metadata"])
        self.assertEqual(images["metadata"]["evidence_level"], "image_metadata_match")
        self.assertIn("image metadata", images["metadata"]["retrieval_basis"])

    def test_image_phrase_does_not_cross_between_ocr_and_ai_description(self):
        material_id = self.material()
        with repository.get_connection() as con:
            con.execute("INSERT INTO image_evidence(id, material_id, evidence_type, source_kind, source_locator, page_ref, page_index, image_path, extraction_method, ocr_text, visual_caption, created_at) VALUES ('image', ?, 'image', 'file_image', 'source.png', '1', 1, 'synthetic.png', 'synthetic', 'Pacific', 'Orchard', '2026-10-05')", (material_id,))
        with patch.object(repository, "ai_embedding_configured", return_value=False):
            result = asyncio.run(repository.search_image_evidence(MultimodalSearchRequest(query="Pacific Orchard", material_id=material_id)))
        self.assertEqual(result["image_results"], [])

    def test_no_generation_context_is_claimed_without_generation(self):
        with patch.object(repository, "ollama_available", AsyncMock(return_value=False)):
            result = asyncio.run(repository.ask_corpus(AskCorpusRequest(question="What is supported?")))
        self.assertNotIn("generation_context", result)
        with patch.object(repository, "ollama_available", AsyncMock(return_value=True)), patch.object(repository, "semantic_search_segments", AsyncMock(return_value={"results": [], "related_observations": []})), patch.object(repository, "search_image_evidence", AsyncMock(return_value={"image_results": []})), patch.object(repository, "request_evidence_answer", AsyncMock(side_effect=AssertionError("No evidence"))):
            result = asyncio.run(repository.ask_corpus(AskCorpusRequest(question="What is supported?")))
        self.assertNotIn("generation_context", result)


if __name__ == "__main__":
    unittest.main()
