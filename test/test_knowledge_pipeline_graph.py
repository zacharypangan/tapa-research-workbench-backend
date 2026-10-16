"""Focused release integrity and review tests; no network or app database."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from app.repository.knowledge_pipeline.graph import GraphValidationError, build_release


class KnowledgePipelineGraphTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ontology = {"version": "1.0", "entity_types": {"Concept": {}}, "predicates": {
            "has_definition": {}, "aligns_reported_concept_with": {"requires_object": True}}}
        self.sources = [
            {"source_id": "primary", "origin": "primary", "source_kind": "pdf",
             "source_version_id": "p:v1", "snapshot_sha256": "a" * 64,
             "seeds": [{"seed_id": "p", "entity_id": "p"}]},
            {"source_id": "web", "origin": "web", "source_kind": "html",
             "source_version_id": "w:v1", "snapshot_sha256": "b" * 64, "seed_ids": ["p"]}]
        self.evidence = [
            {"evidence_id": "ep", "source_id": "primary", "source_version_id": "p:v1",
             "locator": {"page": 1}, "text": "The process is disputed."},
            {"evidence_id": "ew", "source_id": "web", "source_version_id": "w:v1",
             "locator": {"block_id": "b1"}, "quote": "A related process."}]
        self.entities = [
            {"entity_id": "p", "entity_type": "Concept", "label": "Primary process",
             "source_id": "primary", "evidence_ids": ["ep"], "seed_ids": ["p"]},
            {"entity_id": "w", "entity_type": "Concept", "label": "Web process",
             "source_id": "web", "evidence_ids": ["ew"]}]
        self.assertions = [{"assertion_id": "claim", "subject_id": "p", "predicate": "has_definition",
            "value": "Proposed use is disputed.", "source_id": "primary", "evidence_ids": ["ep"],
            "scope": {"case_id": None, "date_basis": "layer, not tool"}, "polarity": "negative"}]
        self.alignments = [{"alignment_id": "alignment", "subject_id": "p", "object_id": "w",
            "predicate": "aligns_reported_concept_with", "evidence_ids": ["ep", "ew"], "seed_ids": ["p"],
            "scope": {"seed_ids": ["p"]}, "polarity": "positive", "status": "candidate"}]

    def tearDown(self):
        self.temp.cleanup()

    def reviews(self, *ids, decision="accept"):
        return [{"candidate_id": cid, "decision": decision, "actor_id": "reviewer",
                 "at": "2026-10-13T10:00:00Z", "rationale": "Reviewed for inclusion."} for cid in ids]

    def build(self, name="release", decisions=None, preview=True):
        return build_release(self.root / name, self.ontology, self.sources, self.entities,
            self.evidence, self.assertions, self.alignments, decisions or [], allow_pending=preview)

    def graph(self, name="release"):
        return json.loads((self.root / name / "graph.json").read_text())

    def test_preview_preserves_polarity_context_and_analytical_layer(self):
        self.build()
        g = self.graph()
        self.assertEqual(g["assertions"][0]["polarity"], "negative")
        self.assertEqual(g["assertions"][0]["scope"], self.assertions[0]["scope"])
        self.assertEqual(g["alignments"][0]["scientific_status"], "project_only")
        self.assertEqual(g["alignments"][0]["review_status"], "needs_specialist_review")
        self.assertIsNone(g["scientific_limits"]["precision"])
        self.assertNotIn("candidate_version", self.entities[0])

    def test_publication_requires_review_and_both_endpoints(self):
        with self.assertRaisesRegex(GraphValidationError, "No accepted"):
            self.build(preview=False)
        with self.assertRaisesRegex(GraphValidationError, "accepted entity endpoints"):
            self.build(decisions=self.reviews("claim"), preview=False)
        with self.assertRaisesRegex(GraphValidationError, "accepted entity endpoints"):
            self.build(decisions=self.reviews("p", "alignment"), preview=False)
        self.assertFalse((self.root / "release").exists())

    def test_partial_review_exports_only_accepted_candidates(self):
        r = self.build(decisions=self.reviews("p", "claim"), preview=False)
        self.assertEqual(r["counts"]["entities"], 1)
        self.assertEqual(r["counts"]["assertions"], 1)
        self.assertEqual(r["counts"]["alignments"], 0)
        self.assertEqual(r["counts"]["specialist_review_decisions"], 0)
        self.assertEqual(self.graph()["assertions"][0]["review_status"], "needs_specialist_review")

    def test_rejection_excludes_dependent_preview_relationships(self):
        r = self.build(decisions=self.reviews("w", decision="reject"))
        self.assertEqual(r["counts"]["entities"], 1)
        self.assertEqual(r["counts"]["alignments"], 0)
        self.assertIn({"candidate_id": "alignment", "reason": "excluded_entity_endpoint"},
                      self.graph()["excluded_candidates"])

    def test_entities_and_assertions_need_own_source_evidence(self):
        self.entities[0]["evidence_ids"] = ["ew"]
        with self.assertRaisesRegex(GraphValidationError, "asserted source"):
            self.build()
        self.entities[0]["evidence_ids"] = ["ep"]
        self.assertions[0]["evidence_ids"] = ["ew"]
        with self.assertRaisesRegex(GraphValidationError, "asserted source"):
            self.build()

    def test_actual_seed_and_primary_endpoint_are_required(self):
        self.sources[1]["seed_ids"] = ["unrelated"]
        with self.assertRaisesRegex(GraphValidationError, "seed is not bound"):
            self.build()
        self.sources[1]["seed_ids"] = ["p"]
        self.alignments[0]["subject_id"], self.alignments[0]["object_id"] = "w", "p"
        with self.assertRaisesRegex(GraphValidationError, "primary subject"):
            self.build()

    def test_alignment_requires_both_evidence_sources_and_consistent_seeds(self):
        self.alignments[0]["evidence_ids"] = ["ep"]
        with self.assertRaisesRegex(GraphValidationError, "both primary and web"):
            self.build()
        self.alignments[0]["evidence_ids"] = ["ep", "ew"]
        self.alignments[0]["scope"]["seed_ids"] = ["other"]
        with self.assertRaisesRegex(GraphValidationError, "Conflicting alignment seed"):
            self.build()

    def test_stale_source_entity_and_review_versions_fail(self):
        self.evidence[0]["source_version_id"] = "p:old"
        with self.assertRaisesRegex(GraphValidationError, "Stale evidence source"):
            self.build()
        self.evidence[0]["source_version_id"] = "p:v1"
        self.assertions[0]["entity_version_ids"] = {"p": "old"}
        with self.assertRaisesRegex(GraphValidationError, "Stale entity version"):
            self.build()
        self.assertions[0].pop("entity_version_ids")
        self.build("preview")
        decision = self.reviews("p")[0]
        decision["candidate_version"] = self.graph("preview")["entities"][0]["candidate_version"]
        self.build("accepted", decisions=[decision], preview=False)
        self.entities[0]["label"] = "Changed interpretation"
        with self.assertRaisesRegex(GraphValidationError, "Stale review"):
            self.build("stale", decisions=[decision], preview=False)

    def test_actor_timezone_and_conflicting_review_validation(self):
        decision = self.reviews("p")[0]
        decision["at"] = "2026-10-13T10:00:00"
        with self.assertRaisesRegex(GraphValidationError, "timezone"):
            self.build(decisions=[decision])
        with self.assertRaisesRegex(GraphValidationError, "Conflicting simultaneous"):
            self.build(decisions=self.reviews("p") + self.reviews("p", decision="reject"))
        with self.assertRaisesRegex(GraphValidationError, "unknown candidate"):
            self.build(decisions=self.reviews("missing"))

    def test_scope_polarity_nonfinite_and_undefined_types_fail(self):
        self.assertions[0]["polarity"] = "maybe"
        with self.assertRaisesRegex(GraphValidationError, "polarity"):
            self.build()
        self.assertions[0]["polarity"] = "negative"
        self.assertions[0]["scope"]["place_id"] = "p"
        with self.assertRaisesRegex(GraphValidationError, "scope reference"):
            self.build()
        self.assertions[0]["scope"].pop("place_id")
        self.assertions[0]["value"] = float("nan")
        with self.assertRaisesRegex(GraphValidationError, "finite JSON"):
            self.build()

    def test_hashes_fresh_directory_and_reified_rdf(self):
        self.sources[0].update(path="/private/never-read.pdf", api_key="not-exported")
        r = self.build()
        self.assertNotIn("path", self.graph()["sources"][0])
        self.assertNotIn("api_key", self.graph()["sources"][0])
        for name, sha in r["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((self.root / "release" / name).read_bytes()).hexdigest(), sha)
        ttl = (self.root / "release" / "graph.ttl").read_text()
        self.assertIn("a kp:SourceClaim", ttl)
        self.assertIn('kp:polarity "negative"', ttl)
        self.assertIn("kp:scopeJSON", ttl)
        self.assertNotIn("kp:has_definition", ttl)
        with self.assertRaises(FileExistsError):
            self.build()

    def test_frozen_compilation_is_byte_deterministic(self):
        self.build("one")
        self.build("two")
        self.assertEqual({p.name: p.read_bytes() for p in (self.root / "one").iterdir()},
                         {p.name: p.read_bytes() for p in (self.root / "two").iterdir()})

    def test_known_acquisition_metadata_survives_without_reading_paths(self):
        provenance = {"source_locator": "https://example.org/final", "requested_url": "https://example.org/start",
                      "snapshot_path": "/unread/runtime/snapshot.html", "representation_path": "/unread/runtime/units.jsonl",
                      "acquisition_method": "http_html", "extractor": "BeautifulSoup/html.parser",
                      "snapshot_hash_basis": "saved decoded HTML", "warnings": ["partial page"],
                      "questions": ["Which material is reported?"], "collection_path": "/unread/runtime/capture"}
        self.sources[1].update(provenance)
        self.build()
        exported = next(s for s in self.graph()["sources"] if s["source_id"] == "web")
        self.assertEqual({key: exported[key] for key in provenance}, provenance)

    def test_scope_records_need_review_and_bind_the_claim_version(self):
        self.ontology["entity_types"]["FindContext"] = {}
        self.entities.append({"entity_id": "place", "entity_type": "FindContext", "label": "Reported context",
                              "source_id": "primary", "evidence_ids": ["ep"]})
        self.assertions[0]["scope"]["place_id"] = "place"
        with self.assertRaisesRegex(GraphValidationError, "accepted entity endpoints"):
            self.build("missing-context", decisions=self.reviews("p", "claim"), preview=False)
        self.build("preview")
        version = self.graph("preview")["assertions"][0]["candidate_version"]
        claim_decision = self.reviews("claim")[0]
        claim_decision["candidate_version"] = version
        self.entities[-1]["label"] = "Corrected context"
        with self.assertRaisesRegex(GraphValidationError, "Stale review"):
            self.build("changed-context", decisions=self.reviews("p", "place") + [claim_decision], preview=False)
        self.build("rejected-context", decisions=self.reviews("place", decision="reject"))
        self.assertEqual(self.graph("rejected-context")["assertions"], [])

    def test_domain_alias_and_dating_assessment_scope_validate(self):
        self.ontology["entity_types"]["DatingAssessment"] = {}
        self.ontology["predicates"]["has_definition"]["domain"] = ["Concept"]
        self.entities.append({"entity_id": "date", "entity_type": "DatingAssessment", "label": "Reported date",
                              "source_id": "primary", "evidence_ids": ["ep"]})
        self.assertions[0]["scope"]["time_id"] = "date"
        self.build("valid")
        self.ontology["predicates"]["has_definition"]["domain"] = ["DatingAssessment"]
        with self.assertRaisesRegex(GraphValidationError, "domain constraint"):
            self.build("invalid")


if __name__ == "__main__":
    unittest.main()
