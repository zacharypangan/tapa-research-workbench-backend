"""Focused offline source-alignment tests, independent of block digest checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .prepare_graph import ValidationError, validate_collection
from .verify_extraction import audit_collection


def digest(value):
    return hashlib.sha256(value).hexdigest()


@unittest.skipUnless(importlib.util.find_spec("bs4"), "Approved isolated BeautifulSoup runtime required")
class FidelityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.collection = self.root / "collection"
        (self.collection / "snapshots").mkdir(parents=True)

    def fixture(self, html, selector=None):
        # The collector supplies candidate rows; audit code never imports these helpers.
        from .extraction import extract_html

        url, stamp = "https://example.org/research", "2026-10-09T03:00:00+00:00"
        parsed = extract_html(html, url, selector)
        sid, content = "web-" + digest(url.encode())[:20], digest(html.encode())
        version = sid + ":" + content
        (self.collection / "snapshots" / (sid + ".html")).write_text(html, encoding="utf-8")
        self.source = {"source_id": sid, "requested_url": url, "retrieved_at": stamp, "depth": 0,
            "discovered_from": None, "review_status": "needs_review", "requested_engine": "static",
            "requested_extractor": "bs4", "status": "collected", "final_url": url, "http_status": 200,
            "browser_http_status": None, "source_version_id": version, "captured_at": stamp,
            "engine": "static", "representation": "http_html", "content_sha256": content,
            "snapshot": "snapshots/" + sid + ".html", "duplicate_of": None,
            "redirect_chain": [{"url": url, "status": 200}], "title": parsed["title"], "language": parsed["language"],
            "canonical_url_metadata": parsed["canonical_url"], "published_at_metadata": parsed["published_at"],
            "extraction_method": parsed["extraction_method"], "text": parsed["text"],
            "extraction_policy": parsed["extraction_policy"], "structured_data": parsed["structured_data"],
            "warnings": parsed["warnings"]}
        self.blocks = [{**block, "evidence_id": version + ":" + block["block_id"], "source_id": sid,
            "source_version_id": version, "source_url": url, "captured_at": stamp,
            "source_content_sha256": content, "review_status": "needs_review"} for block in parsed["blocks"]]
        self.manifest = {"format_version": 1, "created_at": stamp, "successful_sources": 1,
            "attempted_sources": 1, "evidence_blocks": len(self.blocks), "statuses": {"collected": 1},
            "allowed_hosts": ["example.org"], "requested_engine": "static", "requested_extractor": "bs4",
            "seed_urls": [url], "selectors": {"content": selector, "wait": "body"},
            "limits": {"max_pages": 1, "max_depth": 0, "delay": 1.0, "timeout": 20.0,
                       "response_bytes": 3_000_000}, "files": {}}
        self.save()

    def save(self):
        for block in self.blocks:
            block["block_id"] = digest(json.dumps([block["text"], block["locator"]], ensure_ascii=False).encode())
            block["evidence_id"] = self.source["source_version_id"] + ":" + block["block_id"]
        self.source["text"] = "\n\n".join(block["text"] for block in self.blocks)
        self.manifest["evidence_blocks"] = len(self.blocks)
        for name, rows in (("sources.jsonl", [self.source]), ("evidence_blocks.jsonl", self.blocks)):
            data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
            (self.collection / name).write_bytes(data)
            self.manifest["files"][name] = {"sha256": digest(data)}
        (self.collection / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def report(self):
        # Adversarial semantic tampering below remains structurally hash-valid.
        validate_collection(self.collection)
        return audit_collection(self.collection)["collected_source_audits"][0]

    def test_inline_characters_block_boundaries_and_comments_match(self):
        self.fixture('<main><p>Eco<em>nomy</em>, here<br>next<!--omit--></p><p>second</p><nav>noise</nav></main>')
        report = self.report()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["blocks"][0]["source_normalized_text"], "Economy, here next")
        self.assertEqual(report["coverage"]["missing_text_nodes"], 0)
        self.assertEqual(report["coverage"]["overlapping_text_nodes"], 0)

    def test_mixed_container_is_one_truthful_owner(self):
        self.fixture('<main>start<div>inner<em>join</em>.</div>end</main>')
        report = self.report()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["blocks"]), 1)
        self.assertEqual(report["blocks"][0]["source_normalized_text"], "start innerjoin. end")

    def test_table_rows_and_nested_cell_content_are_owned_once(self):
        self.fixture('<main><table><caption>Measures</caption><tr><td>a<table><tr><td>b</td></tr></table></td>'
                     '<td>c</td></tr></table></main>')
        report = self.report()
        self.assertEqual(report["status"], "passed")
        self.assertEqual([block["source_normalized_text"] for block in report["blocks"]], ["Measures", "a b | c"])

    def test_document_locator_and_scope_fallbacks_are_explicit(self):
        self.fixture('alpha <em>beta</em> tail')
        report = self.report()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["blocks"][0]["locator_kind"], "document_root")
        self.fixture('<nav>navigation</nav><main><p>evidence</p></main>', "nav")
        self.assertIn("selected_root_removed_as_noise", self.report()["selection_fallbacks"])
        self.assertEqual(self.report()["status"], "passed")
        self.fixture('<main><p>evidence</p></main>', "[")
        self.assertIn("selector_matched_no_root", self.report()["selection_fallbacks"])
        self.assertEqual(self.report()["status"], "passed")

    def test_disjoint_scope_and_nested_root_deduplication(self):
        self.fixture('<header>outside</header><main><p>alpha</p><p>beta</p></main>', "main, main p")
        report = self.report()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["effective_roots"]), 1)
        self.assertEqual(report["coverage"]["out_of_scope_retained_text_nodes"], 1)

    def test_changed_text_with_recomputed_digests_fails_source_alignment(self):
        self.fixture('<main><p>alpha</p><p>beta</p></main>')
        self.blocks[0]["text"] = "invented interpretation"
        self.save()
        report = self.report()
        self.assertEqual(report["status"], "failed")
        self.assertIn("text_matches_failed", {issue["problem"] for issue in report["issues"]})

    def test_omitted_text_and_reversed_order_are_detected(self):
        self.fixture('<main><p>alpha</p><p>beta</p></main>')
        self.blocks.pop()
        self.save()
        self.assertEqual(self.report()["coverage"]["missing_text_nodes"], 1)
        self.fixture('<main><p>alpha</p><p>beta</p></main>')
        self.blocks.reverse()
        self.save()
        self.assertFalse(self.report()["coverage"]["ownership_in_source_order"])

    def test_coarse_owner_and_descendant_overlap_is_detected(self):
        self.fixture('<main><p>alpha</p><p>beta</p></main>')
        coarse = dict(self.blocks[0], text="alpha beta", locator="main:nth-of-type(1)", tag="main")
        self.blocks.insert(0, coarse)
        self.save()
        report = self.report()
        self.assertEqual(report["coverage"]["overlapping_text_nodes"], 2)
        self.assertTrue(all(row["text_matches"] for row in report["blocks"]))
        self.assertEqual(report["status"], "failed")

    def test_ambiguous_and_out_of_scope_locators_are_detected(self):
        self.fixture('<header><p>outside</p></header><main><p>alpha</p><p>beta</p></main>')
        self.blocks[0]["locator"] = "p"
        self.save()
        self.assertIn("locator_does_not_resolve_once", {issue["problem"] for issue in self.report()["issues"]})
        self.blocks[0].update(locator="header > p", text="outside")
        self.save()
        self.assertIn("within_selected_scope_failed", {issue["problem"] for issue in self.report()["issues"]})

    def test_loading_content_can_pass_fidelity_with_incompleteness_flag(self):
        self.fixture('<main><table><tr><td>LOADING, PLEASE WAIT</td></tr></table></main>')
        report = self.report()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["content_completeness_diagnostics"]["possible_loading_text"])
        self.assertEqual(report["content_completeness_diagnostics"]["rendered_dataset_completeness"], "not_established")

    def test_integrity_failure_precedes_fidelity_and_collection_is_immutable(self):
        self.fixture('<main><p>alpha</p></main>')
        before = {str(path.relative_to(self.collection)): path.read_bytes() for path in self.collection.rglob("*") if path.is_file()}
        audit_collection(self.collection)
        after = {str(path.relative_to(self.collection)): path.read_bytes() for path in self.collection.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        (self.collection / self.source["snapshot"]).write_text("tampered")
        with self.assertRaisesRegex(ValidationError, "snapshot checksum mismatch"):
            audit_collection(self.collection)

    def test_cli_is_no_clobber_and_rejects_output_inside_collection(self):
        self.fixture('<main><p>alpha</p></main>')
        backend = Path(__file__).resolve().parents[2]
        output = self.root / "audit.json"
        command = [sys.executable, "-m", "scripts.tapa_web_pilot.verify_extraction", "--collection",
                   str(self.collection), "--output", str(output)]
        result = subprocess.run(command, cwd=backend, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        original = output.read_bytes()
        result = subprocess.run(command, cwd=backend, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(output.read_bytes(), original)
        inside = self.collection / "audit.json"
        command[-1] = str(inside)
        result = subprocess.run(command, cwd=backend, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(inside.exists())


if __name__ == "__main__":
    unittest.main()
