"""Exact registered-input hashes and GET-only inventories in isolated storage."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api import repository, research_runs
from app.repository import document_graph_inventory as inventory
from app.repository.knowledge_pipeline.registered_extraction import registered_documents
from app.repository.knowledge_pipeline.store import file_hash, read_json


class DocumentGraphInventoryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for symbol, value in {"STORAGE_ROOT": self.root, "FILES_ROOT": self.root / "files",
                              "IMAGES_ROOT": self.root / "images", "DB_PATH": self.root / "repository.sqlite"}.items():
            self.stack.enter_context(patch.object(repository, symbol, str(value)))
        repository.init_repository_db()
        self.material = "material-source"
        self.now = "2026-10-22T00:00:00Z"
        with repository.get_connection() as connection:
            connection.execute("INSERT INTO materials (id,title,source_type,status,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                               (self.material, "石拍 paper", "pdf", "new", self.now, self.now))
        self.file = self.add_file("file-a", "古文.pdf", b"original-source-bytes")
        self.add_file("file-b", "unextracted.txt", b"No registered extraction")
        self.add_segment(self.file, " 第二頁的石拍。\n", page=2)
        self.add_segment(self.file, "第一頁的树皮布。", page=1)
        self.add_segment(self.file, "  \n", page=3)
        self.add_segment(self.file, "Old replaced text", source_kind="historical_file_pdf", page=4)
        self.add_segment(None, "A link-only passage.", source_kind="web", page=5)
        with repository.get_connection() as connection:
            connection.execute("INSERT INTO extraction_runs (id,material_id,include_links,max_link_depth,max_link_pages,extracted_segment_count,discovered_link_count,status,error_message,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                               ("extraction-1", self.material, 0, 0, 0, 2, 0, "ok", None, self.now))
        app = FastAPI()
        app.include_router(research_runs.router, prefix="/api/v1")
        self.client = self.stack.enter_context(TestClient(app))

    def add_file(self, identifier, filename, content, stored=None):
        path = stored or self.root / "files" / identifier
        if stored is None:
            path.write_bytes(content)
        with repository.get_connection() as connection:
            connection.execute("INSERT INTO files (id,material_id,original_filename,stored_path,mime_type,file_size,uploaded_at) VALUES (?,?,?,?,?,?,?)",
                               (identifier, self.material, filename, str(path), "application/octet-stream", len(content), self.now))
        return identifier

    def add_segment(self, file_id, text, source_kind="file_pdf", page=1):
        with repository.get_connection() as connection:
            connection.execute("INSERT INTO extracted_segments (material_id,file_id,source_kind,source_locator,page_ref,page_index,content_text,char_count,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                               (self.material, file_id, source_kind, "古文.pdf", f"page:{page}.1", page, text, len(text), self.now))

    def test_inventory_hash_matches_the_exact_registered_snapshot_bytes(self):
        result = inventory.extraction_inventory(self.material)
        row = next(row for row in result["files"] if row["file_id"] == self.file)
        self.assertTrue(row["eligible"])
        self.assertEqual(row["segment_count"], 2)
        self.assertEqual(row["character_count"], len(" 第二頁的石拍。\n") + len("第一頁的树皮布。"))
        self.assertEqual(result["unassociated_segment_count"], 1)
        self.assertEqual(row["original_sha256"], hashlib.sha256(b"original-source-bytes").hexdigest())
        # The oracle may write a fixture snapshot only after the GET-only check.
        snapshot = registered_documents(self.material, [self.file])[0]["path"]
        self.assertEqual(row["snapshot_sha256"], file_hash(snapshot))
        payload = read_json(snapshot)
        self.assertEqual([segment["page_index"] for segment in payload["segments"]], [1, 2])
        self.assertEqual(payload["extraction_run"]["id"], "extraction-1")
        self.assertEqual(payload["parent_file"]["original_filename"], "古文.pdf")

    def test_get_inventory_creates_no_snapshot_or_source_writes(self):
        before = (self.root / "files" / self.file).read_bytes()
        with patch.object(Path, "write_text", side_effect=AssertionError("GET attempted text write")), \
             patch.object(Path, "write_bytes", side_effect=AssertionError("GET attempted binary write")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("GET attempted artifact directory creation")):
            response = self.client.get(f"/api/v1/research-runs/extraction-inventory/{self.material}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(next(row for row in response.json()["files"] if row["file_id"] == self.file)["eligible"])
        self.assertEqual((self.root / "files" / self.file).read_bytes(), before)
        self.assertFalse((self.root / "graph_representations").exists())
        serialized = json.dumps(response.json(), ensure_ascii=False)
        self.assertNotIn("第一頁的树皮布", serialized)
        self.assertNotIn(str(self.root), serialized)

    def test_empty_current_extraction_does_not_use_historical_or_other_file_text(self):
        result = inventory.extraction_inventory(self.material)
        empty = next(row for row in result["files"] if row["file_id"] == "file-b")
        self.assertFalse(empty["eligible"])
        self.assertEqual(empty["segment_count"], 0)
        self.assertIsNone(empty["snapshot_sha256"])
        self.assertIn("extract this file first", empty["reason"])

    def test_missing_external_and_symlink_escaped_originals_are_ineligible(self):
        self.add_file("missing", "missing.pdf", b"", stored=self.root / "files" / "not-there")
        outside = self.root / "outside.pdf"
        outside.write_bytes(b"external")
        self.add_file("external", "external.pdf", b"external", stored=outside)
        symlink = self.root / "files" / "escaped"
        symlink.symlink_to(outside)
        self.add_file("escaped", "escaped.pdf", b"external", stored=symlink)
        for identifier in ("missing", "external", "escaped"):
            self.add_segment(identifier, "Text that must not authorize missing original bytes.")
        result = inventory.extraction_inventory(self.material)
        for row in result["files"]:
            if row["file_id"] in {"missing", "external", "escaped"}:
                self.assertFalse(row["eligible"])
                self.assertIsNone(row["snapshot_sha256"])
                self.assertNotIn(str(self.root), row["reason"])

    def test_changed_text_and_latest_extraction_identity_change_the_snapshot_hash(self):
        first = inventory.extraction_inventory(self.material)["files"][0]
        self.add_segment(self.file, "A newly extracted passage.", page=3)
        second = inventory.extraction_inventory(self.material)["files"][0]
        self.assertNotEqual(first["snapshot_sha256"], second["snapshot_sha256"])
        with repository.get_connection() as connection:
            connection.execute("INSERT INTO extraction_runs (id,material_id,include_links,max_link_depth,max_link_pages,extracted_segment_count,discovered_link_count,status,error_message,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                               ("extraction-2", self.material, 0, 0, 0, 3, 0, "ok", None, "2026-10-22T01:00:00Z"))
        third = inventory.extraction_inventory(self.material)["files"][0]
        self.assertNotEqual(second["snapshot_sha256"], third["snapshot_sha256"])
        snapshot = registered_documents(self.material, [self.file])[0]["path"]
        self.assertEqual(third["snapshot_sha256"], file_hash(snapshot))

    def test_original_byte_change_is_reflected_in_both_identity_hashes(self):
        first = inventory.extraction_inventory(self.material)["files"][0]
        (self.root / "files" / self.file).write_bytes(b"changed-original-bytes")
        second = inventory.extraction_inventory(self.material)["files"][0]
        self.assertNotEqual(first["original_sha256"], second["original_sha256"])
        self.assertNotEqual(first["snapshot_sha256"], second["snapshot_sha256"])

    def test_unreadable_original_returns_sanitized_ineligibility(self):
        with patch.object(inventory, "file_hash", side_effect=PermissionError("internal storage location must not leak")):
            result = inventory.extraction_inventory(self.material)
        row = next(row for row in result["files"] if row["file_id"] == self.file)
        self.assertFalse(row["eligible"])
        self.assertIn("could not be read", row["reason"])
        self.assertNotIn("internal storage", row["reason"])

    def test_unknown_material_has_no_eligible_input(self):
        with self.assertRaises(HTTPException) as error:
            inventory.extraction_inventory("missing-material")
        self.assertEqual(error.exception.status_code, 404)
