"""Offline integrity/adversarial checks for the graph review-package bridge."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

if __package__:
    from .prepare_graph import ValidationError, prepare, sha256, validate_collection
else:
    from prepare_graph import ValidationError, prepare, sha256, validate_collection


STAMP = "2026-10-09T03:00:00+00:00"


def encoded(value):
    return (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class PrepareGraphTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.collection = self.root / "collection"
        (self.collection / "snapshots").mkdir(parents=True)
        url = "https://example.org/workshop"
        sid = "web-" + sha256(url.encode())[:20]
        html = '<html lang="en"><body><main><p>Reported soaking.</p></main></body></html>'
        content_hash = sha256(html.encode())
        self.html = html
        self.snapshot = self.collection / "snapshots" / f"{sid}.html"
        self.snapshot.write_bytes(html.encode())
        source = {"source_id": sid, "requested_url": url, "retrieved_at": STAMP, "depth": 0,
                  "discovered_from": None, "review_status": "needs_review", "requested_engine": "auto",
                  "requested_extractor": "bs4", "status": "collected", "final_url": url,
                  "http_status": 200, "browser_http_status": None, "source_version_id": sid + ":" + content_hash,
                  "captured_at": STAMP, "engine": "static", "representation": "http_html",
                  "content_sha256": content_hash, "snapshot": f"snapshots/{sid}.html", "duplicate_of": None,
                  "redirect_chain": [{"url": url, "status": 200}], "title": "Reported workshop",
                  "language": "en", "canonical_url_metadata": url, "published_at_metadata": "2020-01-01",
                  "extraction_method": "beautifulsoup_dom_blocks", "text": "Reported soaking.",
                  "structured_data": [{"@type": "Article", "datePublished": "2019-01-01"}], "warnings": []}
        text, locator = source["text"], "html:nth-of-type(1) > body:nth-of-type(1) > main:nth-of-type(1) > p:nth-of-type(1)"
        block_hash = hashlib.sha256(json.dumps([text, locator], ensure_ascii=False).encode()).hexdigest()
        block = {"block_id": block_hash, "text": text, "locator": locator, "tag": "p",
                 "evidence_id": source["source_version_id"] + ":" + block_hash, "source_id": sid,
                 "source_version_id": source["source_version_id"], "source_url": url,
                 "captured_at": STAMP, "source_content_sha256": content_hash, "review_status": "needs_review"}
        skipped_url = "https://example.org/restricted"
        skipped = {"source_id": "web-" + sha256(skipped_url.encode())[:20], "requested_url": skipped_url,
                   "retrieved_at": STAMP, "depth": 0, "discovered_from": None, "review_status": "needs_review",
                   "requested_engine": "auto", "requested_extractor": "bs4", "status": "skipped",
                   "reason": "robots_disallowed"}
        self.sources, self.blocks = [source, skipped], [block]
        self.manifest = {"format_version": 1, "created_at": STAMP, "successful_sources": 1,
                         "attempted_sources": 2, "evidence_blocks": 1,
                         "statuses": {"collected": 1, "skipped": 1}, "allowed_hosts": ["example.org"],
                         "requested_engine": "auto", "requested_extractor": "bs4", "seed_urls": [url, skipped_url],
                         "selectors": {"content": None, "wait": "body"},
                         "limits": {"max_pages": 3, "max_depth": 0, "delay": 1.0, "timeout": 20.0,
                                    "response_bytes": 3_000_000}, "files": {}}
        self.save()

    def save(self):
        for name, rows in (("sources.jsonl", self.sources), ("evidence_blocks.jsonl", self.blocks)):
            data = b"".join(encoded(row) for row in rows)
            (self.collection / name).write_bytes(data)
            self.manifest["files"][name] = {"sha256": sha256(data)}
        (self.collection / "manifest.json").write_bytes(encoded(self.manifest))

    def fails_without_output(self, pattern):
        output = self.root / "new-parent" / "result"
        with self.assertRaisesRegex(ValidationError, pattern):
            prepare(self.collection, output)
        self.assertFalse(output.exists())
        self.assertFalse(output.parent.exists())

    def test_valid_package_preserves_exact_attribution_and_creates_no_claims(self):
        output = self.root / "result"
        manifest = prepare(self.collection, output)
        registry = json.loads((output / "registry_proposal.json").read_text())
        self.assertEqual(manifest["counts"], {"attempts": 2, "registered_source_candidates": 1,
                                           "candidate_blocks": 1, "coverage_units": 2, "entities": 0,
                                           "semantic_evidence": 0, "assertions": 0, "reviews": 0})
        source = registry[0]
        self.assertEqual(source["graph_eligibility"], "quarantine_metadata_only")
        self.assertEqual(source["binding_status"], "UNBOUND")
        self.assertEqual(source["source_kind"], "captured_web_page")
        self.assertEqual(source["published_at_metadata"], "2020-01-01")
        self.assertEqual(source["captured_at"], STAMP)
        self.assertIsNone(source["publication_year"])
        self.assertEqual(Path(source["path"]).read_bytes(), self.html.encode())
        block = read_rows(output / "candidate_blocks.jsonl")[0]
        for key, value in self.blocks[0].items():
            self.assertEqual(block[key], value)
        self.assertEqual(block["verification"], "pending_source_check")
        self.assertEqual(read_rows(output / source["representation_path"]), [block])
        metadata = json.loads((output / source["representation_metadata_path"]).read_text())
        self.assertEqual(metadata, self.sources[0])
        metadata_hash = sha256((output / source["representation_metadata_path"]).read_bytes())
        self.assertEqual(block["source_metadata_sha256"], metadata_hash)
        self.assertEqual(source["related_input_files"], [{"input_id": source["source_id"] + ":capture_metadata",
                                                       "path": str(output / source["representation_metadata_path"]),
                                                       "expected_sha256": metadata_hash}])
        for name in ("entities.jsonl", "evidence.jsonl", "assertions.jsonl", "reviews.jsonl"):
            self.assertEqual((output / name).read_bytes(), b"")
        bindings = json.loads((output / "binding_proposal.json").read_text())
        self.assertTrue(bindings["not_an_accepted_curation_binding"])
        coverage = read_rows(output / "coverage_proposal.jsonl")
        self.assertTrue(all(row["disposition"] == "no_supported_assertion" and not row["assertion_ids"]
                            and not row["evidence_ids"] for row in coverage))
        for kind, ids in source["expected_unit_ids"].items():
            self.assertEqual(set(ids), {row["unit_id"] for row in coverage if row["unit_type"] == kind})
        for name, check in manifest["output_files"].items():
            self.assertEqual(sha256((output / name).read_bytes()), check["sha256"])
        self.assertFalse(manifest["checks"]["source_locator_verified"])
        self.assertFalse(manifest["checks"]["rights_verified"])

    def test_replay_is_deterministic_except_explicit_absolute_path_binding(self):
        first, second = self.root / "first", self.root / "second"
        prepare(self.collection, first)
        prepare(self.collection, second)
        self.assertEqual({str(path.relative_to(first)) for path in first.rglob("*") if path.is_file()},
                         {str(path.relative_to(second)) for path in second.rglob("*") if path.is_file()})
        for path in first.rglob("*"):
            if path.is_file() and path.name not in {"registry_proposal.json", "preparation_manifest.json"}:
                self.assertEqual(path.read_bytes(), (second / path.relative_to(first)).read_bytes())
        first_registry = json.loads((first / "registry_proposal.json").read_text())
        second_registry = json.loads((second / "registry_proposal.json").read_text())
        first_registry[0].pop("path")
        second_registry[0].pop("path")
        first_registry[0]["related_input_files"][0].pop("path")
        second_registry[0]["related_input_files"][0].pop("path")
        self.assertEqual(first_registry, second_registry)

    def test_no_clobber_even_for_existing_empty_directory(self):
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaisesRegex(ValidationError, "existing outputs"):
            prepare(self.collection, output)
        self.assertEqual(list(output.iterdir()), [])

    def test_manifest_file_tamper(self):
        with (self.collection / "sources.jsonl").open("ab") as stream:
            stream.write(b"{}\n")
        self.fails_without_output("manifest checksum mismatch")

    def test_snapshot_tamper(self):
        self.snapshot.write_bytes(b"<p>Changed</p>")
        self.fails_without_output("snapshot checksum mismatch")

    def test_count_and_status_mismatch(self):
        self.manifest["evidence_blocks"] = 2
        self.save()
        self.fails_without_output("count mismatch")
        self.manifest["evidence_blocks"] = 1
        self.manifest["statuses"] = {"collected": 2}
        self.save()
        self.fails_without_output("status counts mismatch")

    def test_unknown_status_cannot_masquerade_as_an_attempt(self):
        self.sources[1]["status"] = "complete"
        self.manifest["statuses"] = {"collected": 1, "complete": 1}
        self.save()
        self.fails_without_output("unknown collection status")

    def test_duplicate_source_id(self):
        self.sources[1] = deepcopy(self.sources[0])
        self.manifest["successful_sources"] = 2
        self.manifest["statuses"] = {"collected": 2}
        self.save()
        self.fails_without_output("duplicate source ID")

    def test_duplicate_evidence_id(self):
        self.blocks.append(deepcopy(self.blocks[0]))
        self.manifest["evidence_blocks"] = 2
        self.save()
        self.fails_without_output("duplicate evidence ID")

    def test_block_text_or_locator_tamper_fails_even_with_updated_manifest(self):
        self.blocks[0]["text"] = "A stronger unsupported statement."
        self.save()
        self.fails_without_output("does not bind exact text and locator")
        self.blocks[0]["text"] = "Reported soaking."
        self.blocks[0]["locator"] = "p:nth-of-type(2)"
        self.save()
        self.fails_without_output("does not bind exact text and locator")

    def test_source_version_and_block_capture_bindings(self):
        self.sources[0]["source_version_id"] += "0"
        self.save()
        self.fails_without_output("source version does not bind")
        self.sources[0]["source_version_id"] = self.blocks[0]["source_version_id"]
        self.blocks[0]["captured_at"] = "2026-10-09T03:00:01+00:00"
        self.save()
        self.fails_without_output("mismatched captured_at")

    def test_failed_and_unknown_source_references(self):
        self.blocks[0]["source_id"] = self.sources[1]["source_id"]
        self.save()
        self.fails_without_output("unknown or unsuccessful source")

    def test_path_traversal_and_absolute_snapshot_are_rejected(self):
        for value in ("../outside.html", str(self.snapshot.absolute()), "snapshots\\page.html"):
            with self.subTest(value=value):
                self.sources[0]["snapshot"] = value
                self.save()
                self.fails_without_output("inside the collection")

    def test_snapshot_symlink_escape_is_rejected(self):
        outside = self.root / "outside.html"
        outside.write_bytes(self.html.encode())
        self.snapshot.unlink()
        self.snapshot.symlink_to(outside)
        self.fails_without_output("symlink escapes")

    def test_input_jsonl_symlink_escape_is_rejected(self):
        path = self.collection / "sources.jsonl"
        outside = self.root / "sources.jsonl"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        self.fails_without_output("symlink escapes")

    def test_missing_metadata_and_naive_timestamp_are_rejected(self):
        captured = self.sources[0].pop("captured_at")
        self.save()
        self.fails_without_output("missing captured_at")
        self.sources[0]["captured_at"] = captured.removesuffix("+00:00")
        self.save()
        self.fails_without_output("timezone required")

    def test_invalid_timestamp_offsets_and_calendar_values_are_rejected(self):
        for value in ("2026-10-09T03:00:00+00:60", "2026-10-09T03:00:00-24:00",
                      "2026-10-09 03:00:00+00:00", "2026-10-09T03:00:00.1234567Z",
                      "2026-02-29T03:00:00Z", "2026-10-09T03:60:00Z"):
            with self.subTest(value=value):
                self.sources[0]["captured_at"] = value
                self.save()
                self.fails_without_output("invalid|complete ISO timestamp")

    def test_metadata_context_changes_invalidate_representation_binding(self):
        first, second = self.root / "original-context", self.root / "changed-context"
        prepare(self.collection, first)
        self.sources[0]["published_at_metadata"] = "2018-01-01"
        self.save()
        prepare(self.collection, second)
        original = json.loads((first / "registry_proposal.json").read_text())[0]
        changed = json.loads((second / "registry_proposal.json").read_text())[0]
        self.assertEqual(original["expected_sha256"], changed["expected_sha256"])
        self.assertNotEqual(original["representation_sha256"], changed["representation_sha256"])
        first_block = read_rows(first / "candidate_blocks.jsonl")[0]
        second_block = read_rows(second / "candidate_blocks.jsonl")[0]
        self.assertEqual(first_block["block_id"], second_block["block_id"])
        self.assertNotEqual(first_block["source_metadata_sha256"], second_block["source_metadata_sha256"])

    def test_existing_kg_builder_rejects_changed_declared_capture_metadata(self):
        tool = Path(__file__).resolve().parents[3] / "SampleReports/tapa_knowledge/tools/build.py"
        if not tool.is_file():
            self.skipTest("Optional existing KG builder is unavailable in this checkout")
        spec = importlib.util.spec_from_file_location("tapa_prepare_integration_fixture", tool)
        pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline)
        output = self.root / "integration-fixture"
        prepare(self.collection, output)
        registry = json.loads((output / "registry_proposal.json").read_text())
        source = registry[0]
        # Accepted bindings exist only inside this temporary adversarial fixture.
        curation = output / "curation"
        curation.mkdir()
        (curation / "ontology.json").write_bytes(encoded({"version": "fixture", "entity_types": {}, "predicates": {}}))
        (curation / "sources.json").write_bytes(encoded(registry))
        (curation / "source_bindings.json").write_bytes(encoded({source["source_id"]: {
            "source_sha256": source["expected_sha256"], "representation_sha256": source["representation_sha256"]}}))
        metadata = Path(source["related_input_files"][0]["path"])
        metadata.write_bytes(metadata.read_bytes() + b" ")
        with self.assertRaisesRegex(pipeline.ValidationError, "changed/missing linked source export"):
            pipeline.build(output, output / "failed-release", "2026-10-09T00:00:00Z")
        self.assertFalse((output / "failed-release").exists())

    def test_actual_engine_and_supplementary_provenance_are_preserved(self):
        self.sources[0].update(engine="playwright", representation="rendered_dom", browser_http_status=200,
                               requested_extractor="trafilatura", article_text="Supplementary wording.",
                               article_text_method="trafilatura_supplementary")
        self.sources[1]["requested_extractor"] = "trafilatura"
        self.manifest["requested_extractor"] = "trafilatura"
        self.save()
        output = self.root / "rendered"
        prepare(self.collection, output)
        lineage = read_rows(output / "source_lineage.jsonl")[0]
        self.assertEqual(lineage["actual_engine"], "playwright")
        self.assertEqual(lineage["supplementary_text_alignment"], "not_source_aligned")
        self.assertEqual(read_rows(output / "candidate_blocks.jsonl")[0]["text"], "Reported soaking.")
        metadata_path = json.loads((output / "registry_proposal.json").read_text())[0]["representation_metadata_path"]
        self.assertEqual(json.loads((output / metadata_path).read_text())["article_text"], "Supplementary wording.")

    def test_engine_representation_mismatch_is_rejected(self):
        self.sources[0]["representation"] = "rendered_dom"
        self.save()
        self.fails_without_output("provenance mismatch")

    def test_equal_snapshots_remain_distinct_source_candidates(self):
        original = self.sources[0]
        duplicate = deepcopy(original)
        url = "https://example.org/reprint"
        sid = "web-" + sha256(url.encode())[:20]
        duplicate.update(source_id=sid, requested_url=url, final_url=url, canonical_url_metadata=url,
                         source_version_id=sid + ":" + original["content_sha256"], duplicate_of=original["source_id"],
                         snapshot=f"snapshots/{sid}.html", redirect_chain=[{"url": url, "status": 200}])
        (self.collection / duplicate["snapshot"]).write_bytes(self.html.encode())
        block = deepcopy(self.blocks[0])
        block.update(source_id=sid, source_version_id=duplicate["source_version_id"], source_url=url,
                     evidence_id=duplicate["source_version_id"] + ":" + block["block_id"])
        self.sources.append(duplicate)
        self.blocks.append(block)
        self.manifest.update(successful_sources=2, attempted_sources=3, evidence_blocks=2,
                             statuses={"collected": 2, "skipped": 1})
        self.manifest["seed_urls"].append(url)
        self.save()
        output = self.root / "duplicate"
        prepare(self.collection, output)
        self.assertEqual(len(json.loads((output / "registry_proposal.json").read_text())), 2)
        duplicates = [row for row in read_rows(output / "source_lineage.jsonl") if row["duplicate_of_collector_source_id"]]
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["duplicate_status"], "equal_snapshot_candidate_only_no_identity_merge")

    def test_invalid_duplicate_reference_is_rejected(self):
        self.sources[0]["duplicate_of"] = self.sources[1]["source_id"]
        self.save()
        self.fails_without_output("duplicate_of does not bind")

    def test_nonfinite_json_and_duplicate_keys_are_rejected(self):
        path = self.collection / "manifest.json"
        path.write_text(path.read_text().replace('"delay": 1.0', '"delay": NaN'))
        self.fails_without_output("Non-finite JSON")
        self.save()
        path.write_text(path.read_text().replace('"delay": 1.0', '"delay": 1e999'))
        self.fails_without_output("Non-finite JSON")
        self.save()
        path.write_text(path.read_text().replace('"format_version": 1', '"format_version": 1, "format_version": 1'))
        self.fails_without_output("Duplicate JSON key")

    def test_allowlist_normalization_matches_collector_without_dns(self):
        self.manifest["allowed_hosts"] = ["EXAMPLE.ORG."]
        self.save()
        self.assertEqual(len(validate_collection(self.collection)["sources"]), 2)

    def test_url_credentials_and_bad_source_identity_are_rejected(self):
        self.sources[0]["requested_url"] = "https://user:secret@example.org/workshop"
        self.save()
        self.fails_without_output("invalid URL")
        self.sources[0]["requested_url"] = "https://example.org/changed"
        self.save()
        self.fails_without_output("does not bind requested URL")

    def test_ordered_block_text_cannot_disagree_with_source(self):
        self.sources[0]["text"] = "Supplementary wording substituted as source text."
        self.save()
        self.fails_without_output("differs from ordered DOM")

    def test_unknown_crawl_parent_and_nonseed_depth_zero_attempt_are_rejected(self):
        self.sources[1].update(depth=1, discovered_from="https://example.org/uncollected")
        self.manifest["limits"]["max_depth"] = 1
        self.save()
        self.fails_without_output("crawl parent is not an earlier collected")
        self.sources[1].update(depth=0, discovered_from=None)
        self.manifest["seed_urls"].pop()
        self.save()
        self.fails_without_output("must bind a seed URL")

    def test_valid_failed_child_retains_parent_provenance(self):
        self.sources[1].update(depth=1, discovered_from=self.sources[0]["final_url"])
        self.manifest["limits"]["max_depth"] = 1
        self.save()
        output = self.root / "child"
        prepare(self.collection, output)
        failed = [row for row in read_rows(output / "attempt_ledger.jsonl") if row["collection_status"] == "skipped"][0]
        self.assertEqual(failed["collector_record"]["discovered_from"], self.sources[0]["final_url"])
        self.assertIsNone(failed["proposed_source_id"])

    def test_malformed_status_engine_and_reference_values_fail_as_validation_errors(self):
        self.sources[1]["status"] = {}
        self.save()
        self.fails_without_output("status must be a string")
        self.sources[1]["status"] = "skipped"
        self.sources[0]["engine"] = []
        self.save()
        self.fails_without_output("invalid actual engine")
        self.sources[0]["engine"] = "static"
        self.blocks[0]["source_id"] = []
        self.save()
        self.fails_without_output("expected a nonempty string")

    def test_all_unsuccessful_collection_is_accounted_for_without_registration(self):
        self.sources = [self.sources[1]]
        self.blocks = []
        self.manifest.update(successful_sources=0, attempted_sources=1, evidence_blocks=0,
                             statuses={"skipped": 1})
        self.save()
        output = self.root / "no-success"
        manifest = prepare(self.collection, output)
        self.assertEqual(manifest["counts"]["registered_source_candidates"], 0)
        self.assertEqual(json.loads((output / "registry_proposal.json").read_text()), [])
        self.assertEqual(len(read_rows(output / "attempt_ledger.jsonl")), 1)

    def test_cli_runs_without_collector_dependencies(self):
        backend = Path(__file__).resolve().parents[2]
        output = self.root / "cli"
        result = subprocess.run([sys.executable, "-m", "scripts.tapa_web_pilot.prepare_graph",
                                 "--collection", str(self.collection), "--output", str(output), "--prefix", "WEB_DEMO"],
                                cwd=backend, check=True, text=True, capture_output=True)
        self.assertEqual(json.loads(result.stdout)["counts"]["assertions"], 0)
        registry = json.loads((output / "registry_proposal.json").read_text())
        self.assertTrue(registry[0]["source_id"].startswith("WEB_DEMO_"))


if __name__ == "__main__":
    unittest.main()
