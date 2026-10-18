import copy
import hashlib
import unittest

from app.repository.knowledge_pipeline.processing import process_response
from scripts.verify_document_graph_run import verify_graph


class GraphVerificationTests(unittest.TestCase):
    def setUp(self):
        text = "😀前文。石拍用于树皮布。"
        unit = {"unit_id": "u", "source_id": "PAPER", "source_version_id": "PAPER:v1",
                "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "locator": {"kind": "pdf_page", "pdf_page": 1}}
        evidence = [{"unit_id": "u", "quote": "石拍用于树皮布。"}]
        response = {"entities": [
            {"local_id": "tool", "entity_type": "Concept", "label_original": "石拍", "evidence": evidence},
            {"local_id": "cloth", "entity_type": "Concept", "label_original": "树皮布", "evidence": evidence}],
            "assertions": [{"local_id": "a", "subject": "tool", "predicate": "related_to", "object": "cloth",
                            "scope": {}, "polarity": "positive", "epistemic_kind": "source_report",
                            "evidence": evidence, "rationale": "Fixture relationship"}]}
        self.graph = process_response(response, [unit], {"entity_types": {"Concept": {}}, "predicates": {"related_to": {}}})
        self.graph["sources"] = [{"source_id": "PAPER", "source_version_id": "PAPER:v1", "source_kind": "file_pdf",
                                  "transcription_coverage": {"readable_units": 1, "total_units": 1}}]
        self.batches = [{"batch_id": "b", "units": [unit]}]

    def test_complete_unicode_slice_and_endpoint_checks(self):
        result = verify_graph(self.graph, self.batches)
        self.assertEqual((result["entities"], result["relationships"], result["quotations"]), (2, 1, 1))
        self.assertEqual(result["readable_pages"], result["total_pages"])
        self.assertTrue(result["integrity_verified"])

    def test_tampered_quote_rejected_even_when_its_hash_is_updated(self):
        changed = copy.deepcopy(self.graph)
        changed["evidence"][0]["text"] = "木拍用于树皮布。"
        changed["evidence"][0]["text_sha256"] = hashlib.sha256(changed["evidence"][0]["text"].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "Unicode source slice"):
            verify_graph(changed, self.batches)

    def test_entity_cannot_borrow_another_sources_quotation(self):
        changed = copy.deepcopy(self.graph)
        changed["sources"].append({"source_id": "OTHER", "source_version_id": "OTHER:v1"})
        changed["entities"][0].update(source_id="OTHER", source_version_id="OTHER:v1")
        with self.assertRaisesRegex(ValueError, "evidence ownership"):
            verify_graph(changed, self.batches)

    def test_dangling_relationship_and_invalid_offsets_rejected(self):
        changed = copy.deepcopy(self.graph)
        changed["assertions"][0]["subject_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "endpoint ownership"):
            verify_graph(changed, self.batches)
        changed = copy.deepcopy(self.graph)
        changed["evidence"][0]["character_locator"]["spans"][0]["start"] = True
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            verify_graph(changed, self.batches)


if __name__ == "__main__":
    unittest.main()
