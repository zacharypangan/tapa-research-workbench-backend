"""Real spawned PDF operations remain safe under concurrent API work."""
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.api import repository as repo
from app.repository import document_representation as representation, pdf_workers


class PdfWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # Parent creates fixtures before any parallel calls; all actual read,
        # image extraction, rendering and native text extraction run in children.
        import fitz
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (240, 240), "white")
        drawing = ImageDraw.Draw(image)
        for x in range(0, 240, 40):
            for y in range(0, 240, 40):
                if (x + y) // 40 % 2:
                    drawing.rectangle((x, y, x + 39, y + 39), fill="black")
        stream = BytesIO()
        image.save(stream, format="PNG")
        self.path = self.root / "source.pdf"
        with fitz.open() as doc:
            page = doc.new_page()
            page.insert_text((72, 72), "Evidence for wooden beaters. " * 5)
            page.insert_image(fitz.Rect(72, 120, 312, 360), stream=stream.getvalue())
            doc.save(self.path)
        self.runtime = {"requested": ["eng"], "selected": [], "warnings": [], "engine_version": "fixture"}

    @classmethod
    def tearDownClass(cls):
        pdf_workers.shutdown_pdf_workers()

    def test_concurrent_text_and_image_operations_preserve_outputs(self):
        with patch.object(repo, "IMAGES_ROOT", str(self.root / "images")), \
                patch.object(representation, "_parse_pdf_native", side_effect=AssertionError("Parent must not parse PDFs")), \
                patch.object(repo, "_extract_images_from_pdf_native", side_effect=AssertionError("Parent must not extract PDF images")), \
                ThreadPoolExecutor(max_workers=4) as callers:
            text_futures = [callers.submit(representation._parse_pdf, self.path, self.runtime) for _ in range(2)]
            image_futures = [callers.submit(repo.extract_images_from_pdf, str(self.path), "material", f"file-{index}", "source.pdf") for index in range(2)]
            for future in text_futures:
                units, warnings, coverage = future.result(timeout=30)
                self.assertIn("Evidence for wooden beaters", units[0]["text"])
                self.assertEqual(coverage["total_units"], 1)
                self.assertFalse(warnings)
            for future in image_futures:
                images, warnings = future.result(timeout=30)
                self.assertFalse(warnings)
                self.assertEqual(len(images), 1)
                self.assertEqual(images[0].page_ref, "page:1.image:1")
                self.assertTrue(Path(images[0].image_path).is_relative_to(self.root / "images"))
                self.assertTrue(Path(images[0].image_path).is_file())

    def test_scanned_pdf_ocr_and_page_cache_survive_process_boundary(self):
        runtime = representation.ocr_runtime("en")
        if not runtime["selected"]:
            self.skipTest("English Tesseract runtime is not installed")
        import fitz
        path = self.root / "scan.pdf"
        with fitz.open() as source, fitz.open() as scanned:
            page = source.new_page()
            page.insert_text((40, 100), "WOODEN BEATER EVIDENCE", fontsize=26)
            image = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            scanned.new_page().insert_image(page.rect, stream=image.tobytes("png"))
            scanned.save(path)
        cache = self.root / "pages"
        units, warnings, coverage = representation._parse_pdf(path, runtime, cache)
        self.assertEqual(units[0]["method"], "tesseract_ocr")
        self.assertIn("BEATER", units[0]["text"].upper())
        self.assertEqual(coverage["ocr_attempted"], 1)
        self.assertTrue((cache / "page-1.json").is_file())
        again, _, coverage = representation._parse_pdf(path, runtime, cache)
        self.assertEqual(units, again)
        self.assertEqual(coverage["ocr_attempted"], 0)

    def test_crashed_worker_is_discarded_and_next_attempt_succeeds(self):
        executor = pdf_workers._get_executor()
        with self.assertRaises(BrokenProcessPool):
            executor.submit(os._exit, 17).result(timeout=20)
        # The first caller observes the failure rather than silently replaying
        # file-writing work; the following durable-job attempt gets a fresh pool.
        with self.assertRaises(BrokenProcessPool):
            pdf_workers.run_pdf_operation("parse", str(self.path), self.runtime, None)
        units, _, _ = pdf_workers.run_pdf_operation("parse", str(self.path), self.runtime, None)
        self.assertIn("Evidence for wooden beaters", units[0]["text"])

    def test_normal_operation_error_does_not_poison_worker(self):
        with self.assertRaisesRegex(ValueError, "Unknown PDF worker operation"):
            pdf_workers.run_pdf_operation("unknown")
        units, _, _ = pdf_workers.run_pdf_operation("parse", str(self.path), self.runtime, None)
        self.assertEqual(units[0]["locator"], "page:1")
