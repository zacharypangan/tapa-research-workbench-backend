"""Focused source/locator/frontier tests; no network or new dependencies."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.repository.knowledge_pipeline import adapters


def sha(data):
    return hashlib.sha256(data).hexdigest()


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def document(self, name, data):
        path = self.base / name
        path.write_bytes(data)
        return path

    def ingest(self, path, directory="ingest"):
        return adapters.ingest_document(path, "P1", "Test paper", self.base / directory)

    def test_text_exact_copy_offsets_and_stable_identity(self):
        raw = "\ufeff  First passage.\nStill first.  \n\n 第二段 \n".encode("utf-8")
        path = self.document("paper.md", raw)
        result = self.ingest(path)
        source, units = result["source"], result["units"]
        self.assertEqual([u["text"] for u in units], ["First passage.\nStill first.", "第二段"])
        decoded = raw.decode("utf-8-sig")
        for unit in units:
            loc = unit["locator"]
            self.assertEqual(decoded[loc["char_start"]:loc["char_end"]], unit["text"])
            self.assertEqual(unit["text_sha256"], sha(unit["text"].encode()))
            self.assertEqual(unit["seed_ids"], [])
        self.assertEqual(Path(source["snapshot_path"]).read_bytes(), raw)
        self.assertEqual(source["raw_sha256"], sha(raw))
        self.assertEqual(source["snapshot_sha256"], sha(raw))
        self.assertEqual(source["representation_sha256"], sha(Path(source["representation_path"]).read_bytes()))
        self.assertEqual(Path(source["snapshot_path"]).stat().st_mode & 0o222, 0)
        repeated = self.ingest(path, "repeat")
        self.assertEqual(result["units"], repeated["units"])
        with self.assertRaisesRegex(ValueError, "fresh output"):
            self.ingest(path)

    def test_json_leaf_pointers_preserve_types_and_escaping(self):
        path = self.document("paper.json", b'{"a/b":{"~name":["tree",null,true]},"empty":[],"blank":""}')
        units = self.ingest(path)["units"]
        self.assertEqual([u["locator"]["json_pointer"] for u in units],
                         ["/a~1b/~0name/0", "/a~1b/~0name/1", "/a~1b/~0name/2", "/empty", "/blank"])
        self.assertEqual([u["locator"]["value_type"] for u in units], ["string", "null", "boolean", "array", "string"])
        self.assertEqual([u["text"] for u in units], ["tree", "null", "true", "[]", ""])

    def test_malformed_inputs_do_not_create_output(self):
        for index, data in enumerate((b'{"same":1,"same":2}', b'{"number":NaN}', b'{')):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.ingest(self.document(f"bad-{index}.json", data), f"bad-{index}")
            self.assertFalse((self.base / f"bad-{index}").exists())
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            self.ingest(self.document("bad.txt", b"\xff"))

    def test_docx_uses_actual_xml_paragraphs_and_table_cells(self):
        stream = BytesIO()
        xml = f'''<w:document xmlns:w="{adapters.WORD_NS}"><w:body>
        <w:p><w:r><w:t>First</w:t><w:tab/><w:t>line</w:t></w:r></w:p>
        <w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p>
        <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Table value</w:t></w:r></w:p>
        <w:p><w:r><w:t>Second cell paragraph</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
        </w:body></w:document>'''
        with ZipFile(stream, "w") as archive:
            archive.writestr("word/document.xml", xml)
        raw = stream.getvalue()
        result = self.ingest(self.document("paper.docx", raw))
        units = result["units"]
        self.assertEqual([u["text"] for u in units], ["First\tline", "Second paragraph", "Table value", "Second cell paragraph"])
        self.assertEqual(units[1]["locator"]["xpath"], "/w:document/w:body/w:p[2]")
        self.assertEqual(units[2]["locator"]["xpath"], "/w:document/w:body/w:tbl[1]/w:tr[1]/w:tc[1]/w:p[1]")
        self.assertEqual(units[3]["locator"]["table_path"], [{"table_index": 1, "row": 1, "cell": 1}])
        self.assertEqual(result["source"]["raw_sha256"], sha(raw))
        self.assertIn("headers", result["source"]["warnings"][0])

    def test_docx_malformed_archive_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Malformed DOCX"):
            self.ingest(self.document("bad.docx", b"not a zip"))

    def test_empty_text_does_not_imply_absence(self):
        result = self.ingest(self.document("empty.txt", b" \n\n "))
        self.assertEqual(result["units"], [])
        self.assertEqual(result["source"]["status"], "unreadable_or_empty")
        self.assertIn("does not establish", result["source"]["warnings"][0])

    def test_scanned_pdf_uses_shared_ocr_transcription_with_page_provenance(self):
        shared = {"representation_id": "shared-v1", "method_version": "transcription-v1",
                  "units": [{"locator": "page:1", "page_index": 1, "text": "石拍用于树皮布。", "method": "tesseract_ocr", "language": "chi_sim", "confidence": .9}],
                  "status": "text_ready", "warnings": [],
                  "coverage": {"unreadable_locators": [], "readable_units": 1}}
        with patch("app.repository.document_representation.extract_document", return_value=shared) as extract:
            result = self.ingest(self.document("scanned.pdf", b"pdf fixture bytes"))
        self.assertEqual(result["source"]["shared_representation_id"], "shared-v1")
        self.assertEqual(result["source"]["transcription_methods"], ["tesseract_ocr"])
        self.assertEqual(result["units"][0]["locator"]["pdf_page"], 1)
        self.assertEqual(result["units"][0]["text"], "石拍用于树皮布。")
        self.assertEqual(result["units"][0]["representation_method"], "tesseract_ocr")
        self.assertTrue(extract.called)

    def test_scanned_pdf_without_ocr_still_reports_missing_pages(self):
        shared = {"representation_id": "shared-empty", "method_version": "transcription-v1", "units": [],
                  "status": "unreadable_or_empty", "warnings": ["OCR language data missing"],
                  "coverage": {"unreadable_locators": ["page:1"], "readable_units": 0}}
        with patch("app.repository.document_representation.extract_document", return_value=shared):
            result = self.ingest(self.document("scanned.pdf", b"pdf fixture bytes"))
        self.assertEqual(result["source"]["unreadable_locators"], [{"kind": "pdf_page", "pdf_page": 1}])
        self.assertEqual(result["source"]["status"], "unreadable_or_empty")
        self.assertIn("No readable text", result["source"]["warnings"][-1])

    def test_legacy_pdf_fallback_discards_partial_failed_parser_output(self):
        def load_page(index):
            if index:
                raise RuntimeError("broken page")
            return SimpleNamespace(get_text=lambda mode: "discarded partial text")
        primary = SimpleNamespace(open=lambda **kwargs: SimpleNamespace(page_count=2, load_page=load_page, close=lambda: None))
        fallback = SimpleNamespace(PdfReader=lambda stream: SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: "Fallback page")]))
        with patch.object(adapters.importlib, "import_module", side_effect=lambda name: primary if name == "fitz" else fallback):
            parts, extractor, warnings, empty = adapters._pdf_parts(b"same bytes")
        self.assertEqual(parts[0][1], "Fallback page")
        self.assertEqual(extractor, "pypdf")
        self.assertIn("RuntimeError", warnings[0])

    def test_frontier_requires_explicit_seeds_and_operator_bounded_options(self):
        valid = [{"url": "https://example.org/item", "seed_ids": ["P1:e1"], "question": "Reported plant material?"}]
        cases = [([{"url": "https://example.org/item", "question": "Question"}], {}),
                 (valid, {"max_depth": 1}), (valid, {"max_depth": False}),
                 (valid, {"browser_binary": "/any/path"}), (valid, {"timeout": float("nan")}),
                 (valid, {"content_selector": []}), (valid, {"allowed_hosts": ["other.org"]}),
                 (valid, {"engine": []})]
        for frontier, options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                adapters.collect_frontier(frontier, self.base / "web", options)
        self.assertFalse((self.base / "web").exists())
        combined = adapters._frontier(valid + [{**valid[0], "url": valid[0]["url"] + "#section", "seed_ids": ["P2:e2"]}])
        self.assertEqual(combined[0]["seed_ids"], ["P1:e1", "P2:e2"])
        self.assertEqual(len(combined), 1)

    def fake_collector(self, extra_url=False, failed=False):
        configs = []

        def collect(config):
            configs.append(config)
            config.output_dir.mkdir()
            rows, blocks = [], []
            urls = config.url + (["https://example.org/unbound"] if extra_url else [])
            for index, url in enumerate(urls):
                sid = "web-" + sha(url.encode())[:20]
                row = {"source_id": sid, "requested_url": url, "status": "skipped" if failed else "collected"}
                if failed:
                    row["reason"] = "robots_disallowed"
                else:
                    raw = b"<main><p>Paper mulberry reported here.</p></main>"
                    filename = sid + ".html"
                    (config.output_dir / filename).write_bytes(raw)
                    version = sid + ":" + sha(raw)
                    row.update(snapshot=filename, content_sha256=sha(raw), source_version_id=version,
                               final_url=url, representation="http_html", extraction_method="bs4_dom_blocks",
                               title="Catalogue record", captured_at="2026-10-13T00:00:00Z")
                    blocks.append({"source_id": sid, "source_version_id": version, "locator": "main > p",
                                   "block_id": "block-1", "text": "Paper mulberry reported here."})
                rows.append(row)
            for name, values in (("sources.jsonl", rows), ("evidence_blocks.jsonl", blocks)):
                (config.output_dir / name).write_text("".join(json.dumps(row) + "\n" for row in values))

        core = SimpleNamespace(Config=lambda **kwargs: SimpleNamespace(**kwargs), collect=collect)
        verifier = SimpleNamespace(audit_collection=lambda path: {"status": "no_collected_sources" if failed else "passed"})
        return core, verifier, configs

    def test_collection_registers_seed_manifest_snapshot_and_representation(self):
        core, verifier, configs = self.fake_collector()
        frontier = [{"url": "https://example.org/item", "seed_ids": ["P1:e1", "P3:a2"],
                     "question": "Which plant is reported?", "selectors": ["main", "article"]}]
        with patch.object(adapters, "_pilot_module", side_effect=lambda name: core if name == "core" else verifier):
            result = adapters.collect_frontier(frontier, self.base / "web", {})
        self.assertEqual(configs[0].engine, "static")
        self.assertEqual(configs[0].max_depth, 0)
        self.assertEqual(configs[0].content_selector, "main, article")
        source = result["sources"][0]
        self.assertEqual(source["origin"], "web")
        self.assertIsNone(source["raw_sha256"])
        self.assertIn("original HTTP bytes not retained", source["snapshot_hash_basis"])
        self.assertEqual(source["snapshot_sha256"], sha(Path(source["snapshot_path"]).read_bytes()))
        self.assertEqual(source["representation_sha256"], sha(Path(source["representation_path"]).read_bytes()))
        self.assertEqual(result["units"][0]["seed_ids"], ["P1:e1", "P3:a2"])
        self.assertEqual(result["units"][0]["locator"]["css_selector"], "main > p")
        manifest = json.loads(Path(result["seed_manifest_path"]).read_text())
        self.assertEqual(manifest["frontier"][0]["questions"], ["Which plant is reported?"])
        self.assertEqual(Path(source["snapshot_path"]).stat().st_mode & 0o222, 0)

    def test_unbound_collected_url_is_rejected(self):
        core, verifier, _ = self.fake_collector(extra_url=True)
        with patch.object(adapters, "_pilot_module", side_effect=lambda name: core if name == "core" else verifier):
            with self.assertRaisesRegex(ValueError, "unbound discovered"):
                adapters.collect_frontier([{"url": "https://example.org/item", "seed_ids": ["P1:e1"], "question": "Why?"}],
                                          self.base / "web", {})

    def test_failed_dom_audit_cannot_supply_units(self):
        core, verifier, _ = self.fake_collector()
        verifier.audit_collection = lambda path: {"status": "failed"}
        with patch.object(adapters, "_pilot_module", side_effect=lambda name: core if name == "core" else verifier):
            with self.assertRaisesRegex(ValueError, "DOM fidelity"):
                adapters.collect_frontier([{"url": "https://example.org/item", "seed_ids": ["P1:e1"], "question": "Why?"}],
                                          self.base / "web", {})
        self.assertEqual((self.base / "web" / "collection-001" / "sources.jsonl").stat().st_mode & 0o222, 0)

    def test_failed_collection_remains_an_attempt_with_seed_attribution(self):
        core, verifier, _ = self.fake_collector(failed=True)
        with patch.object(adapters, "_pilot_module", side_effect=lambda name: core if name == "core" else verifier):
            result = adapters.collect_frontier([{"url": "https://example.org/item", "seed_ids": ["P1:e1"], "question": "Why?"}],
                                              self.base / "web", {})
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["units"], [])
        self.assertEqual(result["attempts"][0]["status"], "skipped")
        self.assertEqual(result["attempts"][0]["reason"], "robots_disallowed")
        self.assertEqual(result["attempts"][0]["seed_ids"], ["P1:e1"])

    def test_real_collector_and_independent_dom_verifier_without_network(self):
        try:
            core = adapters._pilot_module("core")
            adapters.importlib.import_module("bs4")
            adapters.importlib.import_module("httpx")
        except ImportError:
            self.skipTest("Approved optional web-collection runtime is unavailable")

        class FixtureFetcher:
            def __init__(self, config, hosts):
                pass

            def html(self, url):
                return "<html><head><title>Fixture</title></head><body><main><p>Stone beater.</p><p>Reported context.</p></main></body></html>", url, 200, [{"url": url, "status": 200}]

            def close(self):
                pass

        with patch.object(core, "Fetcher", FixtureFetcher):
            result = adapters.collect_frontier([{"url": "https://example.org/item", "seed_ids": ["P4:tool"],
                                                  "question": "Reported tool context?", "selectors": "main"}],
                                              self.base / "web", {"delay": 0})
        self.assertEqual([u["text"] for u in result["units"]], ["Stone beater.", "Reported context."])
        audit = json.loads((self.base / "web" / "dom_audit-001.json").read_text())
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["collected_source_audits"][0]["coverage"]["missing_text_nodes"], 0)

    def test_missing_web_runtime_is_explicit_and_local_ingest_remains_available(self):
        with patch.object(adapters, "_pilot_module", side_effect=ImportError("Web collection requires existing bs4 runtime")):
            with self.assertRaisesRegex(ImportError, "bs4 runtime"):
                adapters.collect_frontier([{"url": "https://example.org/item", "seed_ids": ["P1:e1"], "question": "Why?"}],
                                          self.base / "web", {})
        self.assertFalse((self.base / "web").exists())
        self.assertEqual(self.ingest(self.document("local.txt", b"Still readable"))["units"][0]["text"], "Still readable")


if __name__ == "__main__":
    unittest.main()
