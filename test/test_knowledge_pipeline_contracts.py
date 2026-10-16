"""Cross-module provenance checks using real local acquisition, no network."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.repository.knowledge_pipeline.adapters import ingest_document
from app.repository.knowledge_pipeline.graph import build_release
from app.repository.knowledge_pipeline.processing import process_response
from app.repository.knowledge_pipeline.service import batches_for, default_config


class PipelineContractTests(unittest.TestCase):
    def test_actual_representation_hash_is_distinct_from_unit_text_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.txt"
            path.write_text("A stone beater is described by this source.", encoding="utf-8")
            ingest = ingest_document(path, "DOC_001", "Local source", root / "ingested")
            text_unit = ingest["units"][0]
            ontology = default_config()["ontology"]
            response = {"entities": [{
                "local_id": "form", "entity_type": "ToolForm", "label_original": "stone beater",
                "evidence": [{"unit_id": text_unit["unit_id"], "quote": "stone beater"}],
            }], "assertions": []}
            processed = process_response(response, ingest["units"], ontology)
            self.assertEqual(processed["rejections"], [])
            evidence = processed["evidence"][0]
            self.assertEqual(evidence["unit_text_sha256"], text_unit["text_sha256"])
            self.assertNotEqual(evidence["unit_text_sha256"], ingest["source"]["representation_sha256"])
            summary = build_release(root / "release", ontology, [ingest["source"]],
                                    processed["entities"], processed["evidence"], processed["assertions"], [], [], allow_pending=True)
            self.assertEqual(summary["counts"]["entities"], 1)
            self.assertEqual(summary["counts"]["evidence"], 1)

    def test_split_batch_quotes_retain_parent_locator_and_offsets(self):
        text = "前文。" + "石拍" * 1000
        original = {
            "unit_id": "original", "source_id": "PAPER", "source_version_id": "PAPER:v1",
            "locator": {"kind": "pdf_page", "pdf_page": 12}, "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "seed_ids": [],
        }
        batches = batches_for([original], 1000)
        piece = batches[1]["units"][0]
        ontology = default_config()["ontology"]
        result = process_response({"entities": [{"local_id": "form", "entity_type": "ToolForm", "label_original": "石拍",
                 "evidence": [{"unit_id": piece["unit_id"], "quote": "石拍"}]}], "assertions": []}, [piece], ontology)
        self.assertEqual(result["rejections"], [])
        evidence = result["evidence"][0]
        self.assertEqual(evidence["locator"]["parent_locator"], original["locator"])
        self.assertEqual(evidence["locator"]["parent_unit_id"], "original")
        self.assertEqual(evidence["locator"]["start_char"], 1000)
        self.assertEqual(evidence["character_locator"]["coordinate_system"], "unit_text")
        for span in evidence["character_locator"]["spans"]:
            self.assertEqual(piece["text"][span["start"]:span["end"]], "石拍")


if __name__ == "__main__":
    unittest.main()
