"""Focused checks for retention boundaries, declarations and non-mutation."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("lifecycle_plan.py")
SPEC = importlib.util.spec_from_file_location("tapa_lifecycle_plan", SCRIPT)
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


class LifecyclePlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.registry = self.root / "sources.json"
        self.output = self.root / "plan.json"
        self.source = {
            "source_id": "YOUTUBE_PILOT",
            "path": "/does/not/exist/manifest.json",
            "representation_path": "source_text/YOUTUBE_PILOT.manifest.json",
            "graph_eligibility": "quarantine_metadata_only",
            "expires_at": "2026-11-07T01:53:11.443111+00:00",
            "related_input_files": [{"input_id": "comments", "path": "/does/not/exist/comments.jsonl",
                                     "expected_sha256": "declared_hash_not_verified"}],
        }
        self.registry.write_text(json.dumps([self.source]), encoding="utf-8")

    def run_plan(self, cutoff):
        return planner.write_plan(self.registry, cutoff, self.output)

    def test_utc_offset_and_exact_equality_are_expired(self):
        plan = self.run_plan("2026-11-07T10:53:11.443111+09:00")
        self.assertEqual(plan["sources"][0]["status"], "expired")
        self.assertTrue(plan["sources"][0]["due_at_cutoff"])
        self.assertEqual(plan["sources"][0]["expires_at"], self.source["expires_at"])
        self.assertEqual(plan["summary"]["expired"], 1)
        self.assertEqual([action["timing"] for action in plan["sources"][0]["actions"][1:]],
                         ["due_now", "due_now", "due_now"])

    def test_before_and_after_boundary(self):
        for cutoff, status in (("2026-11-07T01:53:11.443110Z", "active"),
                               ("2026-11-07T01:53:11.443112Z", "expired")):
            with self.subTest(status=status):
                plan = planner.make_plan([self.source], cutoff, "registry.json", "digest")
                self.assertEqual(plan["sources"][0]["status"], status)

    def test_missing_expiry_does_not_invent_a_ttl(self):
        source = {"source_id": "P1", "path": "/not/read.pdf"}
        plan = planner.make_plan([source], "2026-10-09T00:00:00Z", "registry.json", "digest")
        row = plan["sources"][0]
        self.assertEqual(row["status"], "no_declared_expiry")
        self.assertIsNone(row["expires_at"])
        self.assertIsNone(row["deadline_utc"])
        self.assertFalse(row["expires_at_declared"])
        self.assertNotIn("refresh_through_authorized_collector", [a["action"] for a in row["actions"]])

    def test_invalid_cutoff_rejected_without_output(self):
        for cutoff in ("2026-10-09", "2026-10-09T00:00:00", "2026-02-30T00:00:00Z",
                       "2026-10-09T00:00:00+00:60", "2026-10-09T00:00:00+24:00", "nonsense"):
            with self.subTest(cutoff=cutoff):
                with self.assertRaises(ValueError):
                    self.run_plan(cutoff)
                self.assertFalse(self.output.exists())

    def test_invalid_expiry_rejected(self):
        self.source["expires_at"] = "2026-11-07T01:53:11"
        with self.assertRaises(ValueError):
            planner.make_plan([self.source], "2026-10-09T00:00:00Z", "registry.json", "digest")

    def test_registry_keys_numbers_and_ids_rejected(self):
        bad_registries = (
            '[{"source_id":"A","source_id":"B"}]',
            '[{"source_id":"A","nested":{"x":1,"x":2}}]',
            '[{"source_id":"A","value":NaN}]',
            '[{"source_id":"A","value":Infinity}]',
            '[{"source_id":"A","value":1e999}]',
            '[{"source_id":"A"},{"source_id":"A"}]',
            '{}',
        )
        for text in bad_registries:
            with self.subTest(text=text):
                self.registry.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.run_plan("2026-10-09T00:00:00Z")
                self.assertFalse(self.output.exists())

    def test_no_clobber_and_registry_unchanged(self):
        original = self.registry.read_bytes()
        self.output.write_text("existing contents", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_plan("2026-10-09T00:00:00Z")
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing contents")
        self.assertEqual(self.registry.read_bytes(), original)
        self.assertEqual(list(self.root.glob(".%s.*.tmp" % self.output.name)), [])

    def test_write_or_commit_failure_leaves_no_output_or_temporary_file(self):
        original = self.registry.read_bytes()
        for operation in ("fsync", "link"):
            with self.subTest(operation=operation):
                with patch.object(planner.os, operation, side_effect=OSError("injected failure")):
                    with self.assertRaisesRegex(OSError, "injected failure"):
                        self.run_plan("2026-10-09T00:00:00Z")
                self.assertFalse(self.output.exists())
                self.assertEqual(list(self.root.glob(".%s.*.tmp" % self.output.name)), [])
                self.assertEqual(self.registry.read_bytes(), original)

    def test_existing_symlink_is_never_replaced(self):
        missing = self.root / "absent-target"
        self.output.symlink_to(missing)
        with self.assertRaises(FileExistsError):
            self.run_plan("2026-10-09T00:00:00Z")
        self.assertTrue(self.output.is_symlink())
        self.assertEqual(self.output.readlink(), missing)
        self.assertFalse(missing.exists())
        self.assertEqual(list(self.root.glob(".%s.*.tmp" % self.output.name)), [])

    def test_declared_paths_preserved_without_reading_them(self):
        original = self.registry.read_bytes()
        plan = self.run_plan("2026-10-09T00:00:00Z")
        row = plan["sources"][0]
        self.assertEqual(plan["execution_mode"], "dry_run")
        self.assertEqual(row["graph_eligibility"], self.source["graph_eligibility"])
        self.assertFalse(row["eligibility_changed"])
        declared = row["inventory"]["declared_inputs"]
        self.assertEqual([item["declared_input"]["path"] for item in declared],
                         [self.source["path"], self.source["representation_path"],
                          self.source["related_input_files"][0]["path"]])
        self.assertTrue(all(item["presence"] == "not_checked" for item in declared))
        self.assertEqual(row["inventory"]["downstream_completeness"], "incomplete")
        self.assertEqual(self.registry.read_bytes(), original)
        self.assertEqual(plan["registry"]["sha256"], hashlib.sha256(original).hexdigest())
        self.assertIn("Temporary retrieval errors", " ".join(plan["limits"]))

    def test_declared_complete_inventory_requires_all_categories(self):
        self.source["downstream_inventory_complete"] = True
        self.source["downstream_inventory"] = {"releases": ["release.json"]}
        with self.assertRaises(ValueError):
            planner.make_plan([self.source], "2026-10-09T00:00:00Z", "registry.json", "digest")
        self.source["downstream_inventory"] = {category: [] for category in planner.CATEGORIES}
        plan = planner.make_plan([self.source], "2026-10-09T00:00:00Z", "registry.json", "digest")
        self.assertEqual(plan["sources"][0]["inventory"]["downstream_completeness"],
                         "declared_complete_unverified")

    def test_cli_with_actual_registry_and_no_clobber(self):
        actual_registry = SCRIPT.parents[3] / "SampleReports/tapa_knowledge/curation/sources.json"
        self.assertTrue(actual_registry.is_file(), "Current KG registry fixture is required")
        original = actual_registry.read_bytes()
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--registry", str(actual_registry),
             "--as-of", "2026-11-07T01:53:11.443111Z", "--output", str(self.output)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(self.output.read_text(encoding="utf-8"))
        by_id = {source["source_id"]: source for source in plan["sources"]}
        self.assertEqual(by_id["YOUTUBE_PILOT"]["status"], "expired")
        self.assertTrue(by_id["YOUTUBE_PILOT"]["due_at_cutoff"])
        self.assertEqual(by_id["P1"]["status"], "no_declared_expiry")
        self.assertEqual(actual_registry.read_bytes(), original)
        first_output = self.output.read_bytes()
        repeat = subprocess.run(result.args, capture_output=True, text=True)
        self.assertNotEqual(repeat.returncode, 0)
        self.assertEqual(self.output.read_bytes(), first_output)


if __name__ == "__main__":
    unittest.main()
