"""Full-coverage indexing, literal multilingual search and cache regressions."""

import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app.api import repository
from app.repository import embedding_index, image_descriptions, lexical_search
from app.repository.schemas import AskCorpusRequest, MultimodalSearchRequest, SemanticSearchRequest


class RepositoryIndexingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.image_path = root / "test.png"
        self.image_path.write_bytes(b"synthetic image fixture")
        for key, value in {"STORAGE_ROOT": root, "FILES_ROOT": root / "files", "IMAGES_ROOT": root / "images", "DB_PATH": root / "db.sqlite"}.items():
            self.stack.enter_context(patch.object(repository, key, str(value)))
        repository.init_repository_db()
        self.stack.enter_context(patch.object(repository, "ai_configured", return_value=True))
        self.stack.enter_context(patch.object(repository, "request_image_caption", new=AsyncMock(return_value="A visible tool.")))
        self.stack.enter_context(patch.object(repository, "EMBEDDING_PROVIDER", "gemini"))
        self.stack.enter_context(patch.object(repository, "GEMINI_EMBEDDING_MODEL", "test-model"))
        repository._QUERY_EMBEDDINGS.clear()
        repository._READINESS_CACHE.clear()
        with repository.get_connection() as con:
            for material_id in ("one", "two"):
                con.execute("""INSERT INTO materials
                    (id, title, source_type, status, created_at, updated_at)
                    VALUES (?, ?, 'other', 'uploaded', 'test', 'test')""", (material_id, material_id))

    def segment(self, text, material_id="one"):
        with repository.get_connection() as con:
            return con.execute("""INSERT INTO extracted_segments
                (material_id, source_kind, source_locator, page_ref, page_index, content_text, char_count, created_at)
                VALUES (?, 'file_pdf', 'test.pdf', 'page:1', 1, ?, ?, 'test')""", (material_id, text, len(text))).lastrowid

    def image(self, text="stone beater"):
        with repository.get_connection() as con:
            con.execute("""INSERT INTO image_evidence
                (id, material_id, evidence_type, source_kind, source_locator, page_ref, page_index,
                 image_path, extraction_method, ocr_text, visual_caption, created_at)
                VALUES ('image', 'one', 'document_image', 'file_pdf', 'test.pdf', 'page:1', 1,
                    ?, 'test', ?, 'A visible tool.', 'test')""", (str(self.image_path), text))

    def test_full_parent_coverage_and_tail_retrieval_and_replay(self):
        text = "introduction " * 300 + " middle " * 200 + "TAIL_NEEDLE"
        segment_id = self.segment(text)
        async def embed(value, **kwargs):
            return [1., 0.] if "TAIL_NEEDLE" in value else [0., 1.]
        with patch.object(repository, "request_embedding", new=AsyncMock(side_effect=embed)) as mock:
            with repository.get_connection() as con:
                result = asyncio.run(repository.index_missing_segment_embeddings(con))
                self.assertEqual(result["indexed_count"], 1)
                row = con.execute("SELECT * FROM segment_embeddings").fetchone()
                self.assertEqual(row["model"], repository.active_embedding_index_key())
                doc = json.loads(row["embedding_json"])
                self.assertTrue(embedding_index.document_is_current(doc, text, repository.active_embedding_model_name()))
                spans = doc["spans"]
                self.assertEqual(spans[0]["start"], 0)
                self.assertEqual(spans[-1]["end"], len(text))
                self.assertTrue(all(a["end"] >= b["start"] for a, b in zip(spans, spans[1:])))
                self.assertTrue(all(span["end"] - span["start"] <= 2000 for span in spans))
                calls = mock.await_count
                replay = asyncio.run(repository.index_missing_segment_embeddings(con))
                self.assertEqual(replay["indexed_count"], 0)
                self.assertEqual(mock.await_count, calls)
                self.assertEqual(repository.embedding_coverage(con)["embedded_segment_count"], 1)
            search = asyncio.run(repository.semantic_search_segments(SemanticSearchRequest(query="TAIL_NEEDLE", hybrid=False)))
            hit = search["results"][0]
            self.assertEqual(hit["segment_id"], segment_id)
            self.assertEqual(hit["semantic_score"], 1.)
            self.assertIn("TAIL_NEEDLE", hit["matched_text"])
            self.assertGreater(hit["matched_span"]["start"], 2000)

    def test_hash_change_reindexes_even_when_same_id_and_model(self):
        segment_id = self.segment("original")
        with patch.object(repository, "request_embedding", new=AsyncMock(return_value=[1., 0.])) as mock:
            with repository.get_connection() as con:
                asyncio.run(repository.index_missing_segment_embeddings(con))
                con.execute("UPDATE extracted_segments SET content_text = 'changed tail' WHERE id = ?", (segment_id,))
                con.commit()
                self.assertEqual(repository.embedding_coverage(con)["stale_segment_count"], 1)
                result = asyncio.run(repository.index_missing_segment_embeddings(con))
                self.assertEqual(result["indexed_count"], 1)
                self.assertEqual(mock.await_count, 2)

    def test_legacy_vectors_readable_but_not_counted_complete(self):
        segment_id = self.segment("legacy beginning and unembedded tail")
        with repository.get_connection() as con:
            con.execute("INSERT INTO segment_embeddings VALUES (?, 'one', ?, '[1,0]', 'test')", (segment_id, repository.active_embedding_model_name()))
            con.commit()
            self.assertEqual(repository.embedding_coverage(con)["embedded_segment_count"], 0)
        with patch.object(repository, "request_embedding", new=AsyncMock(return_value=[1., 0.])):
            result = asyncio.run(repository.semantic_search_segments(SemanticSearchRequest(query="legacy", hybrid=False)))
            self.assertTrue(result["results"][0]["index_legacy"])

    def test_failures_are_per_item_and_retried_without_losing_successes(self):
        failed = self.segment("FAIL_ITEM")
        successful = self.segment("success")
        async def embed(value, **kwargs):
            if value == "FAIL_ITEM":
                raise HTTPException(status_code=502, detail="temporary failure")
            return [1., 0.]
        with patch.object(repository, "request_embedding", new=AsyncMock(side_effect=embed)) as mock:
            with repository.get_connection() as con:
                result = asyncio.run(repository.index_missing_segment_embeddings(con))
                self.assertEqual(result["indexed_count"], 1)
                self.assertEqual(result["failed_ids"], [failed])
                self.assertEqual(result["pending_count"], 1)
                self.assertEqual(mock.await_count, 4)
                self.assertEqual(con.execute("SELECT segment_id FROM segment_embeddings").fetchone()[0], successful)

    def test_skip_failed_ids_allows_later_healthy_batches_before_partial_failure(self):
        failed = self.segment("FAIL_FIRST")
        healthy = [self.segment(f"healthy-{index}") for index in range(5)]
        async def embed(value, **kwargs):
            if value == "FAIL_FIRST":
                raise HTTPException(status_code=502, detail="unavailable item")
            return [1., 0.]
        with patch.object(repository, "request_embedding", new=AsyncMock(side_effect=embed)) as mock:
            with repository.get_connection() as con:
                first = asyncio.run(repository.index_missing_segment_embeddings(con, limit=2))
                self.assertEqual(first["failed_ids"], [failed])
                self.assertEqual(first["indexed_count"], 1)
                for _ in range(2):
                    result = asyncio.run(repository.index_missing_segment_embeddings(con, limit=2, skip_ids={failed}))
                    self.assertEqual(result["indexed_count"], 2)
                result = asyncio.run(repository.index_missing_segment_embeddings(con, limit=2, skip_ids={failed}))
                self.assertEqual(result["indexed_count"], 0)
                self.assertEqual(result["pending_count"], 1)
                self.assertEqual([row[0] for row in con.execute("SELECT segment_id FROM segment_embeddings ORDER BY segment_id")], healthy)
                self.assertEqual(mock.await_count, 8)

    def test_duplicate_spans_share_calls_and_provider_concurrency_is_bounded(self):
        for i in range(8):
            self.segment("duplicate" if i < 4 else f"different-{i}")
        active = maximum = 0
        async def embed(value, **kwargs):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(.005)
            active -= 1
            return [1., 0.]
        with patch.object(repository, "request_embedding", new=AsyncMock(side_effect=embed)) as mock:
            with repository.get_connection() as con:
                self.assertEqual(asyncio.run(repository.index_missing_segment_embeddings(con))["indexed_count"], 8)
            self.assertEqual(mock.await_count, 5)
            self.assertGreater(maximum, 1)
            self.assertLessEqual(maximum, 3)

    def test_long_image_description_tail_is_indexed_and_searchable(self):
        self.image("beginning " * 500 + "TAIL_NEEDLE")
        async def embed(value, **kwargs):
            return [1., 0.] if "TAIL_NEEDLE" in value else [0., 1.]
        with patch.object(repository, "is_informative_image", return_value=True), patch.object(repository, "request_embedding", new=AsyncMock(side_effect=embed)):
            with repository.get_connection() as con:
                result = asyncio.run(repository.index_missing_image_embeddings(con))
                self.assertEqual(result["indexed_count"], 1)
                self.assertEqual(result["coverage"]["embedded_image_count"], 1)
            result = asyncio.run(repository.search_image_evidence(MultimodalSearchRequest(query="TAIL_NEEDLE")))
            self.assertEqual(result["image_results"][0]["semantic_score"], 1.)
            self.assertGreater(result["image_results"][0]["matched_span"]["start"], 2000)

    def test_uninformative_images_are_retained_skipped_and_not_repeated(self):
        self.image()
        with patch.object(repository, "is_informative_image", return_value=False) as informative, patch.object(repository, "request_embedding", new=AsyncMock()) as embed:
            with repository.get_connection() as con:
                result = asyncio.run(repository.index_missing_image_embeddings(con))
                self.assertEqual(result["skipped_count"], 1)
                self.assertEqual(result["pending_count"], 0)
                self.assertEqual(con.execute("SELECT count(*) FROM image_evidence").fetchone()[0], 1)
                self.assertEqual(asyncio.run(repository.index_missing_image_embeddings(con))["processed_image_count"], 0)
                informative.assert_called_once()
                embed.assert_not_awaited()

    def test_query_embedding_reused_by_text_and_image_channels(self):
        with patch.object(repository, "request_embedding", new=AsyncMock(return_value=[1., 0.])) as embed:
            async def search():
                await asyncio.gather(repository.semantic_search_segments(SemanticSearchRequest(query="same query")),
                                     repository.search_image_evidence(MultimodalSearchRequest(query="same query")))
                await repository.semantic_search_segments(SemanticSearchRequest(query="same query"))
            asyncio.run(search())
            self.assertEqual(embed.await_count, 1)

    def test_provider_readiness_is_cached_and_distinct_from_generation(self):
        with patch.object(repository, "_probe_embedding_readiness", new=AsyncMock(return_value=(True, "ready"))) as probe:
            async def checks():
                self.assertTrue((await repository.check_embedding_readiness())[0])
                self.assertTrue((await repository.check_embedding_readiness())[0])
            asyncio.run(checks())
            self.assertEqual(probe.await_count, 1)
        with patch.object(repository, "check_embedding_readiness", new=AsyncMock(return_value=(True, "ready"))), patch.object(repository, "check_ollama_generation_readiness", new=AsyncMock(return_value=(False, "generation failed"))), patch.object(repository, "request_evidence_answer", new=AsyncMock()) as answer:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(repository.ask_corpus(AskCorpusRequest(question="what is here?")))
            self.assertEqual(raised.exception.status_code, 503)
            answer.assert_not_awaited()

    def test_cjk_short_terms_and_filters_use_and_semantics(self):
        wanted = self.segment("广西地区发现石拍用于制作树皮布。")
        self.segment("广西地区只有石拍。")
        self.segment("广西地区发现石拍用于制作树皮布。", "two")
        with repository.get_connection() as con:
            self.assertEqual([r["id"] for r in lexical_search.search_rows(con, "石拍 树皮", "one")], [wanted])
            self.assertEqual(lexical_search.search_rows(con, "石拍%", "one"), [])

    def test_malformed_literal_does_not_raise_and_advanced_is_validated(self):
        self.segment("stone beater")
        with repository.get_connection() as con:
            self.assertEqual(len(lexical_search.search_rows(con, '"stone', "one")), 1)
            self.assertEqual(lexical_search.search_rows(con, 'stone OR missing', "one"), [])
            with self.assertRaises(ValueError):
                lexical_search.search_rows(con, '"stone', "one", advanced=True)
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(repository.search_extracted_text(q="  ", material_id=None, limit=20, advanced=False))
        self.assertEqual(raised.exception.status_code, 400)

    def test_hybrid_search_includes_unembedded_literal_evidence(self):
        wanted = self.segment("广西地区发现石拍。")
        with patch.object(repository, "request_embedding", new=AsyncMock(return_value=[1., 0.])):
            result = asyncio.run(repository.semantic_search_segments(SemanticSearchRequest(query="石拍")))
            self.assertEqual(result["results"][0]["segment_id"], wanted)
            self.assertEqual(result["ranking_method"], "reciprocal_rank_fusion_k60")
            self.assertEqual(result["results"][0]["semantic_rank"], None)
            self.assertGreater(result["results"][0]["hybrid_score"], 0.)

    def test_observation_retained_historical_rows_do_not_reenter_current_indexes(self):
        current = self.segment("current stone beater")
        historical = self.segment("old stone beater")
        self.image()
        with repository.get_connection() as con:
            con.execute("UPDATE extracted_segments SET source_kind = 'historical_file_pdf' WHERE id = ?", (historical,))
            con.execute("UPDATE image_evidence SET source_kind = 'historical_file_pdf'")
            con.commit()
            self.assertEqual([r["id"] for r in lexical_search.search_rows(con, "stone beater")], [current])
            self.assertEqual(repository.embedding_coverage(con)["segment_count"], 1)
            self.assertEqual(repository.embedding_coverage(con)["image_evidence_count"], 0)
            with patch.object(repository, "request_embedding", new=AsyncMock(return_value=[1., 0.])):
                self.assertEqual(asyncio.run(repository.index_missing_segment_embeddings(con))["indexed_count"], 1)
                result = asyncio.run(repository.semantic_search_segments(SemanticSearchRequest(query="stone", hybrid=False)))
                self.assertEqual([r["segment_id"] for r in result["results"]], [current])

    def test_missing_image_source_is_failed_instead_of_permanently_skipped(self):
        self.image_path.unlink()
        self.image()
        with patch.object(repository, "is_informative_image", return_value=False):
            with repository.get_connection() as con:
                result = asyncio.run(repository.index_missing_image_embeddings(con))
                self.assertEqual(result["failed_ids"], ["image"])
                self.assertEqual(result["skipped_count"], 0)
                self.assertEqual(result["pending_count"], 1)

    def test_concurrent_image_annotation_change_is_not_overwritten(self):
        self.image()
        with repository.get_connection() as con:
            con.execute("UPDATE image_evidence SET visual_caption = ''")
            con.commit()
        async def caption(*args):
            with repository.get_connection() as con:
                con.execute("UPDATE image_evidence SET visual_caption = 'Human correction'")
            return "Generated description"
        with patch.object(repository, "is_informative_image", return_value=True), patch.object(repository, "request_image_caption", new=AsyncMock(side_effect=caption)):
            with repository.get_connection() as con:
                result = asyncio.run(repository.index_missing_image_embeddings(con))
                self.assertEqual(result["outcomes"][0]["status"], "stale")
                self.assertEqual(con.execute("SELECT visual_caption FROM image_evidence").fetchone()[0], "Human correction")

    def test_empty_caption_keeps_ocr_searchable_and_reuses_vector_on_bounded_retries(self):
        self.image(text="")
        with repository.get_connection() as con:
            con.execute("UPDATE image_evidence SET visual_caption = ''")
            con.commit()
        with patch.object(repository, "is_informative_image", return_value=True), patch.object(repository, "ocr_image_file", return_value="stone beater OCR"), patch.object(repository, "request_image_caption", new=AsyncMock(return_value="")) as caption, patch.object(repository, "request_embedding", new=AsyncMock(return_value=[1., 0.])) as embed:
            with repository.get_connection() as con:
                for attempt in range(3):
                    result = asyncio.run(repository.index_missing_image_embeddings(con, force=attempt > 0))
                    self.assertEqual(result["failed_ids"], ["image"])
                    self.assertEqual(result["failed_count"], 1)
                    self.assertEqual(result["indexed_count"], 0)
                    self.assertEqual(result["pending_count"], 1)
                    self.assertTrue(result["outcomes"][0]["ocr_searchable"])
                    self.assertEqual(result["outcomes"][0]["error_category"], "no_usable_observable_labels")
                    self.assertEqual(embed.await_count, 1)
                self.assertEqual(caption.await_count, 3)
                self.assertEqual(con.execute("SELECT ocr_text FROM image_evidence").fetchone()[0], "stone beater OCR")
                coverage = repository.embedding_coverage(con)
                self.assertEqual(coverage["embedded_image_count"], 0)
                self.assertEqual(coverage["missing_image_count"], 1)
                self.assertEqual(coverage["ocr_only_searchable_image_count"], 1)
                held = asyncio.run(repository.index_missing_image_embeddings(con, skip_ids={"image"}))
                self.assertEqual(held["processed_image_count"], 0)
                self.assertEqual(caption.await_count, 3)
            found = asyncio.run(repository.search_image_evidence(MultimodalSearchRequest(query="stone beater"), query_embedding=[1., 0.]))
            self.assertEqual([item["image_id"] for item in found["image_results"]], ["image"])
            self.assertEqual(found["image_results"][0]["description_status"], "failed")
            self.assertIn("OCR/observation text only", found["image_results"][0]["retrieval_basis"])

    def test_existing_ocr_only_vector_does_not_reembed_on_failed_caption_and_recovers(self):
        self.image("stone beater OCR")
        with repository.get_connection() as con:
            con.execute("UPDATE image_evidence SET visual_caption = ''")
            con.commit()
            row = repository._image_index_rows(con)[0]
            with patch.object(repository, "request_embedding", new=AsyncMock(return_value=[1., 0.])):
                doc = asyncio.run(repository._embed_span_document("stone beater OCR", {}, asyncio.Semaphore(1)))
            inputs = image_descriptions.description_input(row["image_path"], repository.get_image_context_text(con, row), repository.OLLAMA_VISION_MODEL)
            doc.update(description_key=inputs["key"], description_caption_sha256=embedding_index.content_hash(""))
            con.execute("INSERT INTO image_embeddings VALUES ('image', 'one', ?, ?, ?, 'previous')",
                        (repository.active_embedding_index_key(), repository.MULTIMODAL_METHOD_VERSION, json.dumps(doc)))
            con.commit()
        with patch.object(repository, "is_informative_image", return_value=True), patch.object(repository, "request_image_caption", new=AsyncMock(side_effect=["", "Recovered labels"])) as caption, patch.object(repository, "request_embedding", new=AsyncMock(return_value=[1., 0.])) as embed:
            with repository.get_connection() as con:
                first = asyncio.run(repository.index_missing_image_embeddings(con))
                self.assertEqual(first["failed_ids"], ["image"])
                self.assertTrue(first["outcomes"][0]["vector_reused"])
                self.assertEqual(embed.await_count, 0)
                recovered = asyncio.run(repository.index_missing_image_embeddings(con))
                self.assertEqual(recovered["failed_count"], 0)
                self.assertEqual(recovered["indexed_count"], 1)
                self.assertEqual(recovered["pending_count"], 0)
                self.assertEqual(recovered["coverage"]["embedded_image_count"], 1)
                self.assertEqual(recovered["coverage"]["ocr_only_searchable_image_count"], 0)
                self.assertEqual(embed.await_count, 1)
                self.assertEqual(caption.await_count, 2)
                replay = asyncio.run(repository.index_missing_image_embeddings(con))
                self.assertEqual(replay["processed_image_count"], 0)
                self.assertEqual(caption.await_count, 2)

    def test_ocr_vector_survives_vision_model_change_but_not_changed_image_or_text(self):
        self.image("stone beater OCR")
        with repository.get_connection() as con:
            con.execute("UPDATE image_evidence SET visual_caption = ''")
            con.commit()
        with patch.object(repository, "is_informative_image", return_value=True), \
                patch.object(repository, "request_image_caption", new=AsyncMock(return_value="")), \
                patch.object(repository, "request_embedding", new=AsyncMock(return_value=[1., 0.])) as embed:
            with repository.get_connection() as con:
                asyncio.run(repository.index_missing_image_embeddings(con))
                with patch.object(repository, "OLLAMA_VISION_MODEL", "corrected-vision-model"):
                    self.assertEqual(repository.embedding_coverage(con)["ocr_only_searchable_image_count"], 1)
                    found = asyncio.run(repository.search_image_evidence(MultimodalSearchRequest(query="stone beater"), query_embedding=[1., 0.]))
                    self.assertEqual([item["image_id"] for item in found["image_results"]], ["image"])
                    self.assertEqual(found["image_results"][0]["description_status"], "failed")
                    con.execute("UPDATE image_evidence SET ocr_text = 'corrected transcription'")
                    con.commit()
                    self.assertEqual(repository.embedding_coverage(con)["ocr_only_searchable_image_count"], 0)
                    con.execute("UPDATE image_evidence SET ocr_text = 'stone beater OCR'")
                    con.commit()
                    self.image_path.write_bytes(b"replaced image bytes")
                    self.assertEqual(repository.embedding_coverage(con)["ocr_only_searchable_image_count"], 0)
            self.assertEqual(embed.await_count, 1)

    def test_caption_failure_preserves_old_caption_but_does_not_search_it(self):
        self.image("stone beater")
        with repository.get_connection() as con:
            con.execute("UPDATE image_evidence SET visual_caption = 'UNTRUSTED_OLD_CAPTION'")
            con.commit()
        with patch.object(repository, "is_informative_image", return_value=True), patch.object(repository, "request_image_caption", new=AsyncMock(return_value=" ")), patch.object(repository, "request_embedding", new=AsyncMock(return_value=[1., 0.])):
            with repository.get_connection() as con:
                result = asyncio.run(repository.index_missing_image_embeddings(con))
                self.assertEqual(result["failed_count"], 1)
                self.assertEqual(con.execute("SELECT visual_caption FROM image_evidence").fetchone()[0], "UNTRUSTED_OLD_CAPTION")
            found = asyncio.run(repository.search_image_evidence(MultimodalSearchRequest(query="UNTRUSTED_OLD_CAPTION"), query_embedding=[0., 1.]))
            self.assertEqual(found["image_results"], [])
            found = asyncio.run(repository.search_image_evidence(MultimodalSearchRequest(query="stone"), query_embedding=[1., 0.]))
            self.assertEqual(found["image_results"][0]["visual_caption"], "")
            self.assertEqual(found["image_results"][0]["previous_visual_caption"], "UNTRUSTED_OLD_CAPTION")

    def test_caption_failure_without_readable_ocr_is_explicit_and_does_not_embed(self):
        self.image("")
        with repository.get_connection() as con:
            con.execute("UPDATE image_evidence SET visual_caption = ''")
            con.commit()
        with patch.object(repository, "is_informative_image", return_value=True), patch.object(repository, "ocr_image_file", return_value=""), patch.object(repository, "request_image_caption", new=AsyncMock(return_value="")), patch.object(repository, "request_embedding", new=AsyncMock()) as embed:
            with repository.get_connection() as con:
                result = asyncio.run(repository.index_missing_image_embeddings(con))
                self.assertEqual(result["failed_ids"], ["image"])
                self.assertFalse(result["outcomes"][0]["ocr_searchable"])
                self.assertEqual(result["pending_count"], 1)
                self.assertEqual(con.execute("SELECT COUNT(*) FROM image_embeddings").fetchone()[0], 0)
                embed.assert_not_awaited()

    def test_safe_http_failure_diagnostics_and_manifest_category_are_retained(self):
        self.segment("provider fails")
        with patch.object(repository, "request_embedding", new=AsyncMock(side_effect=HTTPException(status_code=401, detail="SECRET_TOKEN must not appear"))):
            with repository.get_connection() as con:
                result = asyncio.run(repository.index_missing_segment_embeddings(con))
                outcome = result["outcomes"][0]
                self.assertEqual(outcome["http_status"], 401)
                self.assertEqual(outcome["error_category"], "authentication")
                self.assertFalse(outcome["retryable"])
                self.assertIn("HTTP 401", outcome["error"])
                self.assertNotIn("SECRET_TOKEN", json.dumps(outcome))
        self.image("stone beater")
        with repository.get_connection() as con:
            row = repository._image_index_rows(con)[0]
            inputs = image_descriptions.description_input(row["image_path"], repository.get_image_context_text(con, row), repository.OLLAMA_VISION_MODEL)
        image_descriptions.save_record(repository.STORAGE_ROOT, {**inputs, "status": "failed", "failure_category": "authentication", "attempts": [{"http_status": 403, "raw_response": "PRIVATE_PROVIDER_BODY"}]})
        with patch.object(repository, "is_informative_image", return_value=True), patch.object(repository, "request_image_caption", new=AsyncMock(return_value="")), patch.object(repository, "request_embedding", new=AsyncMock(return_value=[1., 0.])):
            with repository.get_connection() as con:
                result = asyncio.run(repository.index_missing_image_embeddings(con))
                outcome = result["outcomes"][0]
                self.assertEqual(outcome["http_status"], 403)
                self.assertEqual(outcome["error_category"], "authentication")
                self.assertFalse(outcome["retryable"])
                self.assertNotIn("PRIVATE_PROVIDER_BODY", json.dumps(outcome))

    def test_provider_caption_exception_keeps_useful_ocr_with_safe_status(self):
        self.image("stone beater OCR")
        with patch.object(repository, "is_informative_image", return_value=True), patch.object(repository, "request_image_caption", new=AsyncMock(side_effect=HTTPException(status_code=429, detail="SECRET_PROVIDER_DETAIL"))), patch.object(repository, "request_embedding", new=AsyncMock(return_value=[1., 0.])):
            with repository.get_connection() as con:
                result = asyncio.run(repository.index_missing_image_embeddings(con))
                outcome = result["outcomes"][0]
                self.assertEqual(result["failed_ids"], ["image"])
                self.assertTrue(outcome["ocr_searchable"])
                self.assertEqual(outcome["http_status"], 429)
                self.assertEqual(outcome["error_category"], "rate_limit")
                self.assertTrue(outcome["retryable"])
                self.assertNotIn("SECRET_PROVIDER_DETAIL", json.dumps(outcome))

    def test_caption_recipe_refreshes_once_and_reuses_unchanged_vectors(self):
        self.image()
        caption = repository.request_image_caption
        with patch.object(repository, "is_informative_image", return_value=True), patch.object(repository, "request_embedding", new=AsyncMock(return_value=[1., 0.])) as embed:
            with repository.get_connection() as con:
                self.assertEqual(asyncio.run(repository.index_missing_image_embeddings(con))["indexed_count"], 1)
                self.assertEqual(caption.await_count, 1)
                self.assertEqual(asyncio.run(repository.index_missing_image_embeddings(con))["indexed_count"], 0)
                self.assertEqual(caption.await_count, 1)
                with patch.object(repository, "OLLAMA_VISION_MODEL", "new-vision-model"):
                    self.assertEqual(repository.embedding_coverage(con)["stale_image_count"], 1)
                    self.assertEqual(asyncio.run(repository.index_missing_image_embeddings(con))["indexed_count"], 1)
                    self.assertEqual(caption.await_count, 2)
                    self.assertEqual(embed.await_count, 1)
                    self.segment("New same-page context")
                    self.assertEqual(asyncio.run(repository.index_missing_image_embeddings(con))["indexed_count"], 1)
                    self.assertEqual(caption.await_count, 3)
                    self.assertEqual(embed.await_count, 1)

    def test_generated_answer_flags_and_citation_bounds(self):
        evidence = [{"content_text": "Source evidence", "material_title": "one", "citation": {"material_id": "one"}}]
        for generated, expected_valid in (("Supported [1].", True), ("Bad source [99].", False), ("No numbered citation.", True)):
            with self.subTest(generated=generated), ExitStack() as stack:
                stack.enter_context(patch.object(repository, "check_embedding_readiness", new=AsyncMock(return_value=(True, "ready"))))
                stack.enter_context(patch.object(repository, "check_ollama_generation_readiness", new=AsyncMock(return_value=(True, "ready"))))
                stack.enter_context(patch.object(repository, "query_embedding_for_search", new=AsyncMock(return_value=[1., 0.])))
                stack.enter_context(patch.object(repository, "semantic_search_segments", new=AsyncMock(return_value={"results": evidence, "related_observations": []})))
                stack.enter_context(patch.object(repository, "search_image_evidence", new=AsyncMock(return_value={"image_results": []})))
                stack.enter_context(patch.object(repository, "request_evidence_answer", new=AsyncMock(return_value=generated)))
                result = asyncio.run(repository.ask_corpus(AskCorpusRequest(question="question " * 90)))
                self.assertEqual(result["answer_validation"]["citation_indices_valid"], expected_valid)
                self.assertTrue(result["answer_validation"]["review_required"])
                self.assertEqual(result["answer_validation"]["claim_support"], "not_automatically_verified")
                if expected_valid:
                    self.assertEqual(result["answer_origin"], "model_generated")
                else:
                    self.assertTrue(result["answer_validation"]["generated_answer_withheld"])
                    self.assertNotIn("[99]", result["answer"])

    def test_invalid_vectors_and_oversized_raw_inputs_rejected(self):
        for vector in ([0., 0.], [float("nan")], [True], []):
            with self.assertRaises(ValueError):
                embedding_index.validated_vector(vector)
        with self.assertRaises(HTTPException):
            asyncio.run(repository.request_embedding("x" * 2001))


if __name__ == "__main__":
    unittest.main()
