"""Integrity, locator, accounting and replay checks for JSON snapshot extraction."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import extract_json
except ImportError:
    import extract_json


class JsonExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = self.root / "input.json"
        self.output = self.root / "package"
        self.url = "https://museum.example/objects/1"
        self.captured = "2026-10-09T14:00:00+09:00"

    def save(self, value):
        self.snapshot.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return hashlib.sha256(self.snapshot.read_bytes()).hexdigest()

    def run_extract(self, digest, pointer=None, output=None):
        return extract_json.extract(self.snapshot, self.url, self.captured, digest,
                                    output or self.output, pointer)

    def rows(self, name):
        return [json.loads(line) for line in (self.output / name).read_text(encoding="utf-8").splitlines()]

    def test_every_scalar_and_empty_container_is_typed_and_html_is_verbatim(self):
        value = {"string": "<p lang='fj'>masi &amp; tapa</p>", "integer": 2,
                 "number": 2.5, "boolean": False, "null": None,
                 "empty_object": {}, "empty_array": [], "nested": {"value": "tapa"}}
        digest = self.save(value)
        original = self.snapshot.read_bytes()
        manifest = self.run_extract(digest)
        fields = {row["locator"]["json_pointer"]: row for row in self.rows("candidate_fields.jsonl")}
        self.assertEqual(len(fields), 8)
        for key, expected in (("string", "string"), ("integer", "integer"), ("number", "number"),
                              ("boolean", "boolean"), ("null", "null"),
                              ("empty_object", "empty_object"), ("empty_array", "empty_array")):
            self.assertEqual(fields["/" + key]["value_type"], expected)
        self.assertEqual(fields["/string"]["value"], value["string"])
        self.assertEqual(fields["/string"]["interpretation"], "uninterpreted_json_field")
        self.assertEqual((self.output / "snapshot.json").read_bytes(), original)
        self.assertEqual(self.snapshot.read_bytes(), original)
        self.assertEqual(manifest["counts"]["scalar_fields"], 6)
        self.assertEqual(manifest["counts"]["empty_container_fields"], 2)
        self.assertEqual(self.rows("record_inventory.jsonl")[0]["structural_leaf_count"], 8)
        self.assertEqual(os.stat(self.output / "snapshot.json").st_mode & 0o777, 0o400)
        for name in ("entities.jsonl", "evidence.jsonl", "assertions.jsonl", "reviews.jsonl"):
            self.assertEqual((self.output / name).read_bytes(), b"")

    def test_rfc6901_escaping_and_nested_record_associations(self):
        value = {"data": [{"a/b": {"~key": ["first", {}]}, "": "empty key"}, {"v": None}],
                 "total": 2}
        digest = self.save(value)
        manifest = self.run_extract(digest, "/data")
        fields = {row["locator"]["json_pointer"]: row for row in self.rows("candidate_fields.jsonl")}
        self.assertEqual(set(fields), {"/data/0/a~1b/~0key/0", "/data/0/a~1b/~0key/1",
                                      "/data/0/", "/data/1/v", "/total"})
        for pointer, row in fields.items():
            self.assertEqual(extract_json.resolve_pointer(value, pointer), row["value"])
        self.assertIsNone(fields["/total"]["record_id"])
        self.assertNotEqual(fields["/data/0/"]["record_id"], fields["/data/1/v"]["record_id"])
        objects = {row["json_pointer"]: row for row in self.rows("object_associations.jsonl")}
        self.assertEqual(objects["/data/0/a~1b"]["parent_object_association_id"],
                         objects["/data/0"]["object_association_id"])
        self.assertEqual(fields["/data/0/a~1b/~0key/0"]["object_association_id"],
                         objects["/data/0/a~1b"]["object_association_id"])
        self.assertEqual(manifest["counts"]["records"], 2)
        self.assertEqual(manifest["counts"]["fields_outside_selected_records"], 1)
        self.assertEqual(len(self.rows("coverage.jsonl")), 7)

    def test_empty_root_and_root_record_list_pointer(self):
        digest = self.save({})
        manifest = self.run_extract(digest)
        self.assertEqual(self.rows("candidate_fields.jsonl")[0]["locator"]["json_pointer"], "")
        self.assertEqual(manifest["counts"]["structural_leaves"], 1)
        digest = self.save([{"x": 1}, {"x": 2}])
        manifest = self.run_extract(digest, "", self.root / "list-package")
        self.assertEqual(manifest["counts"]["records"], 2)

    def test_selected_empty_record_list_still_accounts_for_empty_container(self):
        digest = self.save({"data": [], "total": 0})
        manifest = self.run_extract(digest, "/data")
        self.assertEqual(manifest["counts"]["records"], 0)
        self.assertEqual(manifest["counts"]["structural_leaves"], 2)
        self.assertEqual(manifest["counts"]["fields_outside_selected_records"], 2)

    def test_invalid_pointer_or_record_selection_creates_no_output(self):
        digest = self.save({"data": [{"x": 1}], "scalar": 0, "mixed": [{}, 2]})
        for pointer in ("data", "#/data", "/missing", "/data/~2", "/data/00", "/data/-",
                        "/data/2", "/scalar/field", "/scalar", "/mixed", ""):
            with self.subTest(pointer=pointer), self.assertRaises(ValueError):
                self.run_extract(digest, pointer)
            self.assertFalse(self.output.exists())

    def test_duplicate_keys_and_nonfinite_numbers_are_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"nested":{"x":1,"x":2}}',
                    b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":1e999}'):
            with self.subTest(raw=raw):
                self.snapshot.write_bytes(raw)
                with self.assertRaises(ValueError):
                    self.run_extract(hashlib.sha256(raw).hexdigest())
                self.assertFalse(self.output.exists())

    def test_snapshot_tamper_and_invalid_attribution_create_no_output(self):
        digest = self.save({"x": 1})
        self.snapshot.write_text('{"x":2}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.run_extract(digest)
        digest = hashlib.sha256(self.snapshot.read_bytes()).hexdigest()
        for url, timestamp in (("file:///etc/passwd", self.captured),
                               ("https://user:secret@museum.example/", self.captured),
                               (self.url, "2026-10-09T05:00:00"),
                               (self.url, "2026-10-09T05:00:00+00:60")):
            with self.subTest(url=url, timestamp=timestamp), self.assertRaises(ValueError):
                extract_json.extract(self.snapshot, url, timestamp, digest, self.output)
            self.assertFalse(self.output.exists())

    def test_no_clobber_existing_directory_file_or_symlink(self):
        digest = self.save({"x": 1})
        for kind in ("directory", "file", "symlink"):
            output = self.root / kind
            if kind == "directory":
                output.mkdir()
            elif kind == "file":
                output.write_text("keep", encoding="utf-8")
            else:
                output.symlink_to(self.root / "absent")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.run_extract(digest, output=output)
            self.assertTrue(output.exists() or output.is_symlink())
        self.assertEqual((self.root / "file").read_text(), "keep")
        self.assertEqual(list(self.root.glob(".extract-json-*")), [])

    def test_commit_failure_cleans_output_and_temporary_directory(self):
        digest = self.save({"x": 1})
        with patch.object(extract_json.os, "replace", side_effect=OSError("injected commit failure")):
            with self.assertRaises(OSError):
                self.run_extract(digest)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".extract-json-*")), [])

    def test_replay_is_byte_identical_and_all_output_hashes_match(self):
        digest = self.save({"z": [{"a": True}], "a": "樹皮布"})
        manifest = self.run_extract(digest)
        replay = self.root / "replay"
        self.run_extract(digest, output=replay)
        self.assertEqual(sorted(path.name for path in self.output.iterdir()),
                         sorted(path.name for path in replay.iterdir()))
        for path in self.output.iterdir():
            self.assertEqual(path.read_bytes(), (replay / path.name).read_bytes())
        for name, metadata in manifest["files"].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), metadata["sha256"])

    def test_standalone_cli_and_offset_metadata(self):
        digest = self.save({"data": [{"x": "tapa"}]})
        result = subprocess.run([
            sys.executable, str(Path(extract_json.__file__)), "--snapshot", str(self.snapshot),
            "--source-url", self.url, "--captured-at", self.captured,
            "--expected-sha256", digest, "--record-list-pointer", "/data",
            "--output", str(self.output),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((self.output / "source_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["captured_at"], self.captured)
        self.assertEqual(metadata["captured_at_utc"], "2026-10-09T05:00:00+00:00")
        self.assertEqual(metadata["graph_eligibility"], "quarantine_metadata_only")
        self.assertEqual(json.loads(result.stdout)["counts"]["semantic_assertions"], 0)


if __name__ == "__main__":
    unittest.main()
