"""Offline fixtures for shared OCR/transcription and non-destructive refresh."""
import asyncio
from io import BytesIO
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import httpx

from app.api import repository as repo
from app.repository import document_representation as representation
from app.repository.schemas import ExtractRequest, ExtractionResult, ImageEvidenceInput, MaterialCreate, SegmentInput


class DocumentRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        # OCR unit fixtures exercise the native parser with local mocks. Real
        # spawned-process behavior is covered independently in test_pdf_workers.
        self.stack.enter_context(patch.object(representation, "_parse_pdf", side_effect=representation._parse_pdf_native))

    def pdf(self, name="scan.pdf", text=None):
        import fitz
        path = self.root / name
        with fitz.open() as doc:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
            else:
                from PIL import Image, ImageDraw
                image = Image.new("RGB", (800, 200), "white")
                ImageDraw.Draw(image).text((20, 20), "Synthetic scan", fill="black")
                stream = BytesIO()
                image.save(stream, format="PNG")
                page.insert_image(page.rect, stream=stream.getvalue())
            doc.save(path)
        return path

    def test_scanned_page_ocr_cached_and_located(self):
        path = self.pdf()
        runtime = {"requested": ["eng"], "selected": ["eng"], "warnings": [], "engine_version": "synthetic"}
        with patch.object(representation, "ocr_runtime", return_value=runtime), patch.object(representation, "ocr_image", return_value={"text": "Synthetic wooden beater", "confidence": .96, "language": "eng", "warnings": []}) as ocr:
            result = representation.extract_document(path, path.name, cache_root=self.root / "cache")
            again = representation.extract_document(path, path.name, cache_root=self.root / "cache")
        self.assertEqual(result, again)
        self.assertEqual(ocr.call_count, 1)
        self.assertEqual(result["units"][0]["locator"], "page:1")
        self.assertEqual(result["units"][0]["method"], "tesseract_ocr")
        self.assertEqual(result["units"][0]["confidence"], .96)

    def test_missing_chinese_language_is_reported_without_english_fallback(self):
        with patch("pytesseract.get_languages", return_value=["eng"]), patch("pytesseract.get_tesseract_version", return_value="5.0"):
            runtime = representation.ocr_runtime("Chinese")
        self.assertEqual(runtime["selected"], [])
        self.assertIn("chi_sim", runtime["warnings"][0])
        with patch.object(representation, "ocr_runtime", return_value=runtime), patch.object(representation, "ocr_image") as ocr:
            result = representation.extract_document(self.pdf("中文.pdf"), "中文.pdf", cache_root=self.root / "cache", languages="Chinese")
        ocr.assert_not_called()
        self.assertEqual(result["status"], "unreadable_or_empty")
        self.assertTrue(any("chi_sim" in warning for warning in result["warnings"]))

    def test_ocr_language_and_timeout_passed_and_failure_explained(self):
        runtime = {"selected": ["chi_sim", "eng"], "requested": ["chi_sim", "eng"], "warnings": []}
        with patch("pytesseract.image_to_data", side_effect=RuntimeError("timeout")) as call:
            result = representation.ocr_image(object(), runtime, timeout=7)
        self.assertEqual(call.call_args.kwargs["lang"], "chi_sim+eng")
        self.assertEqual(call.call_args.kwargs["timeout"], 7)
        self.assertIn("timed out", result["warnings"][0])

    def test_unknown_metadata_uses_installed_cjk_and_kana_filename_prefers_japanese(self):
        with patch("pytesseract.get_languages", return_value=["eng", "chi_sim", "chi_tra", "jpn"]), patch("pytesseract.get_tesseract_version", return_value="5.0"):
            self.assertEqual(representation.ocr_runtime(filename="jaas062002.pdf")["selected"], ["chi_sim", "chi_tra", "eng", "jpn"])
            self.assertEqual(representation.ocr_runtime(filename="ミュージアムDX実践ガイド.pdf")["selected"], ["eng", "jpn"])

    def test_docx_is_xml_text_not_zip_bytes_and_binary_rejected(self):
        path = self.root / "document.docx"
        with ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>真实材料</w:t></w:r></w:p></w:body></w:document>')
        result = representation.extract_document(path, path.name, cache_root=self.root / "cache")
        self.assertEqual(result["units"][0]["text"], "真实材料")
        self.assertEqual(result["units"][0]["locator"], "paragraph:1")
        result = representation.extract_document(path, "pretend.txt", cache_root=self.root / "cache")
        self.assertEqual(result["units"], [])
        self.assertIn("Binary signature", result["warnings"][0])

    def test_content_change_creates_new_immutable_representation(self):
        path = self.root / "document.txt"
        path.write_text("first")
        first = representation.extract_document(path, path.name, cache_root=self.root / "cache")
        path.write_text("second")
        second = representation.extract_document(path, path.name, cache_root=self.root / "cache")
        self.assertNotEqual(first["representation_id"], second["representation_id"])
        self.assertEqual(len(list((self.root / "cache" / "artifacts").glob("*.json"))), 2)

    def test_partial_ocr_retry_reuses_successful_pages_only(self):
        import fitz
        path = self.pdf()
        two_pages = self.root / "two.pdf"
        with fitz.open(path) as source, fitz.open() as document:
            document.insert_pdf(source)
            document.insert_pdf(source)
            document.save(two_pages)
        runtime = {"requested": ["eng"], "selected": ["eng"], "warnings": [], "engine_version": "synthetic"}
        success = {"text": "Readable synthetic page", "confidence": .95, "language": "eng", "warnings": []}
        failure = {"text": "", "confidence": None, "language": "eng", "warnings": ["OCR timed out"]}
        with patch.object(representation, "ocr_runtime", return_value=runtime), patch.object(representation, "ocr_image", side_effect=[success, failure, success]) as ocr:
            first = representation.extract_document(two_pages, two_pages.name, cache_root=self.root / "cache")
            second = representation.extract_document(two_pages, two_pages.name, cache_root=self.root / "cache")
        self.assertEqual(first["status"], "partial_text")
        self.assertEqual(second["status"], "text_ready")
        self.assertNotEqual(first["representation_id"], second["representation_id"])
        self.assertEqual(ocr.call_count, 3)


class RepositoryRefreshTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for symbol, value in {"DB_PATH": self.root / "repo.sqlite", "FILES_ROOT": self.root / "files", "IMAGES_ROOT": self.root / "images"}.items():
            self.stack.enter_context(patch.object(repo, symbol, str(value)))
        repo.init_repository_db()
        self.material = asyncio.run(repo.create_material(MaterialCreate(title="Synthetic fixture")))["id"]
        with repo.get_connection() as con:
            con.execute("INSERT INTO files(id,material_id,original_filename,stored_path,mime_type,file_size,uploaded_at) VALUES(?,?,?,?,?,?,?)", ("f1", self.material, "source.pdf", "unused", "application/pdf", 1, repo.now_iso()))

    def run_extract(self, result, images=(), force=False):
        with patch.object(repo, "extract_text_from_file", return_value=result), patch.object(repo, "extract_images_from_file", return_value=(list(images), [])), patch.object(repo, "dedupe_image_inputs", side_effect=lambda values: (values, 0)):
            return asyncio.run(repo.extract_material_text(self.material, None, ExtractRequest(include_links=False), force=force))

    def source(self, text="Original readable text"):
        return ExtractionResult([SegmentInput("file_pdf", "source.pdf", "page:1.1", 1, text)], [])

    def test_failed_refresh_preserves_segments_observation_and_history(self):
        self.run_extract(self.source())
        with repo.get_connection() as con:
            segment_id = con.execute("SELECT id FROM extracted_segments").fetchone()[0]
            con.execute("INSERT INTO observations(id,material_id,source_segment_id,observation_type,observed_text,created_at,updated_at) VALUES(?,?,?,?,?,?,?)", ("obs", self.material, segment_id, "term", "readable", repo.now_iso(), repo.now_iso()))
        with self.assertRaises(repo.HTTPException) as exception:
            self.run_extract(ExtractionResult([], ["parser failed"]), force=True)
        self.assertEqual(exception.exception.status_code, 422)
        with repo.get_connection() as con:
            self.assertEqual(con.execute("SELECT content_text FROM extracted_segments").fetchone()[0], "Original readable text")
            self.assertEqual(con.execute("SELECT source_segment_id FROM observations").fetchone()[0], segment_id)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0], 1)

    def test_successful_refresh_reuses_ids_and_keeps_referenced_old_evidence(self):
        self.run_extract(self.source())
        with repo.get_connection() as con:
            segment_id = con.execute("SELECT id FROM extracted_segments").fetchone()[0]
            con.execute("INSERT INTO observations(id,material_id,source_segment_id,observation_type,observed_text,created_at,updated_at) VALUES(?,?,?,?,?,?,?)", ("obs", self.material, segment_id, "term", "readable", repo.now_iso(), repo.now_iso()))
        self.run_extract(self.source(), force=True)
        with repo.get_connection() as con:
            self.assertEqual(con.execute("SELECT id FROM extracted_segments").fetchone()[0], segment_id)
        self.run_extract(self.source("Corrected readable text"), force=True)
        with repo.get_connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM extracted_segments").fetchone()[0], 2)
            self.assertEqual(con.execute("SELECT source_segment_id FROM observations").fetchone()[0], segment_id)
            self.assertEqual(con.execute("SELECT source_kind FROM extracted_segments WHERE id = ?", (segment_id,)).fetchone()[0], "historical_file_pdf")
        extracted = asyncio.run(repo.get_material_extracted(self.material, limit=200))
        self.assertEqual(len(extracted["segments"]), 1)
        self.assertEqual(len(extracted["historical_segments"]), 1)
        self.assertTrue(extracted["historical_segments"][0]["is_historical"])

    def test_partial_refresh_retains_prior_unreadable_page(self):
        source = self.source()
        source.segments.append(SegmentInput("file_pdf", "source.pdf", "page:2.1", 2, "Prior second page"))
        self.run_extract(source)
        partial = self.source()
        partial.warnings.append("Incomplete transcription: one unreadable page")
        result = self.run_extract(partial, force=True)
        self.assertTrue(any("Prior evidence" in item for item in result["warnings"]))
        with repo.get_connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM extracted_segments").fetchone()[0], 2)

    def test_1001_images_no_post_commit_payload_failure(self):
        images = [ImageEvidenceInput("f1", "document_image", "file_pdf_image", "source.pdf", f"page:1.image:{index}", 1, "unused.png", "image/png", 200, 200, "synthetic") for index in range(1001)]
        result = self.run_extract(self.source(), images)
        self.assertEqual(result["image_evidence_count"], 1001)
        with repo.get_connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM image_evidence").fetchone()[0], 1001)

    def test_refresh_keeps_referenced_original_image_bytes_and_marks_history(self):
        old_path, new_path = self.root / "original.png", self.root / "replacement.png"
        old_path.write_bytes(b"original accepted image")
        new_path.write_bytes(b"replacement figure with similar visual fingerprint")
        def image(path):
            return ImageEvidenceInput("f1", "document_image", "file_pdf_image", "source.pdf", "page:1.image:1", 1, str(path), "image/png", 200, 200, "synthetic", fingerprint="aa")
        self.run_extract(self.source(), [image(old_path)])
        with repo.get_connection() as con:
            old_id = con.execute("SELECT id FROM image_evidence").fetchone()[0]
            con.execute("INSERT INTO observations(id,material_id,source_image_id,observation_type,observed_text,created_at,updated_at) VALUES(?,?,?,?,?,?,?)", ("obs", self.material, old_id, "motif", "accepted figure", repo.now_iso(), repo.now_iso()))
        self.run_extract(self.source(), [image(new_path)], force=True)
        self.assertEqual(old_path.read_bytes(), b"original accepted image")
        self.assertTrue(new_path.exists())
        with repo.get_connection() as con:
            self.assertEqual(con.execute("SELECT source_image_id FROM observations").fetchone()[0], old_id)
            self.assertEqual(con.execute("SELECT source_kind FROM image_evidence WHERE id = ?", (old_id,)).fetchone()[0], "historical_file_pdf_image")
            self.assertEqual(con.execute("SELECT COUNT(*) FROM image_evidence WHERE source_kind NOT GLOB 'historical_*'").fetchone()[0], 1)

    def test_image_only_failed_refresh_keeps_prior_evidence(self):
        path = self.root / "original.png"
        path.write_bytes(b"image-only accepted evidence")
        image = ImageEvidenceInput("f1", "document_image", "file_pdf_image", "source.pdf", "page:1.image:1", 1, str(path), "image/png", 200, 200, "synthetic")
        self.run_extract(ExtractionResult([], []), [image])
        with self.assertRaises(repo.HTTPException) as exception:
            self.run_extract(ExtractionResult([], ["parser failed"]), force=True)
        self.assertEqual(exception.exception.status_code, 422)
        self.assertTrue(path.exists())
        with repo.get_connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM image_evidence").fetchone()[0], 1)

    def test_image_context_never_uses_another_file_same_page(self):
        self.run_extract(self.source())
        with repo.get_connection() as con:
            unrelated = {"material_id": self.material, "file_id": "other", "source_locator": "another.pdf", "page_index": 1, "page_ref": "page:1"}
            self.assertEqual(repo.get_image_context_text(con, unrelated), "")
            own = {**unrelated, "file_id": "f1", "source_locator": "source.pdf"}
            self.assertIn("Original readable", repo.get_image_context_text(con, own))


class RepositoryWebCollectionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(repo, "STORAGE_ROOT", str(self.root)))
        self.requests = []

    def collect(self, handler, url="https://example.com/", max_depth=1):
        original = httpx.AsyncClient
        def transport(request):
            self.requests.append(str(request.url))
            return handler(request)
        with patch.object(repo.httpx, "AsyncClient", side_effect=lambda **kwargs: original(transport=httpx.MockTransport(transport), **kwargs)), patch("scripts.tapa_web_pilot.browser_engines.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]):
            return asyncio.run(repo.crawl_and_extract_links(url, max_depth, 3))

    def test_private_root_and_redirect_rejected_before_request(self):
        segments, logs = self.collect(lambda request: httpx.Response(200), url="http://127.0.0.1/")
        self.assertEqual(self.requests, [])
        self.assertEqual(segments, [])
        self.assertIn("non-public", logs[0]["status"])
        def handler(request):
            return httpx.Response(404) if request.url.path == "/robots.txt" else httpx.Response(302, headers={"location": "http://127.0.0.1/private"})
        segments, logs = self.collect(handler)
        self.assertEqual(segments, [])
        self.assertFalse(any("127.0.0.1" in url for url in self.requests))
        self.assertIn("outside allowed_hosts", logs[0]["status"])

    def test_stream_byte_cap_and_binary_content_type_are_explicit(self):
        for headers, content, reason in [({"content-type": "text/html"}, b"x" * 3_000_001, "response_too_large"), ({"content-type": "application/pdf"}, b"%PDF", "unsupported_content_type")]:
            def handler(request):
                return httpx.Response(404) if request.url.path == "/robots.txt" else httpx.Response(200, headers=headers, content=content)
            segments, logs = self.collect(handler)
            self.assertEqual(segments, [])
            self.assertIn(reason, logs[0]["status"])

    def test_same_host_only_and_hashed_observed_snapshot(self):
        def handler(request):
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b'<p>Synthetic barkcloth source.</p><a href="https://elsewhere.example/">Outside</a>')
        segments, logs = self.collect(handler)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].source_locator, "https://example.com/")
        self.assertFalse(any("elsewhere" in url for url in self.requests))
        manifest_path = next((self.root / "web_snapshots").glob("*/manifest.json"))
        import json
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["source_locator"], segments[0].source_locator)
        self.assertEqual(manifest["acquisition_method"], "httpx_stream_v1")
        self.assertEqual(logs[0]["snapshot_id"], manifest["snapshot_id"])
        self.assertEqual(len(manifest["source_sha256"]), 64)

    def test_robots_disallowed_path_not_requested(self):
        def handler(request):
            return httpx.Response(200, headers={"content-type": "text/plain"}, text="User-agent: *\nDisallow: /blocked")
        segments, logs = self.collect(handler, url="https://example.com/blocked")
        self.assertEqual(self.requests, ["https://example.com/robots.txt"])
        self.assertEqual(segments, [])
        self.assertEqual(logs[0]["status"], "robots_disallowed")


if __name__ == "__main__":
    unittest.main()
