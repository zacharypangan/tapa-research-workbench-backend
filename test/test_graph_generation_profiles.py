"""Guidance changes new-run configuration without weakening source constraints."""
import copy
from pathlib import Path
import tempfile
import unittest

from app.repository.graph_generation_profiles import (
    GENERATION_PROFILE_VERSION,
    generation_ontology,
    prepare_generation_config,
)
from app.repository.knowledge_pipeline.ontologies import ontology_template
from app.repository.knowledge_pipeline.processing import extraction_prompt, extraction_schema
from app.repository.knowledge_pipeline.service import validate_config, verify_implementation
from app.repository.knowledge_pipeline.store import RunStore, digest, read_json


class GraphGenerationProfilesTest(unittest.TestCase):
    def test_both_templates_retain_semantics_and_output_schema(self):
        for template_id in ("general", "tapa"):
            with self.subTest(template=template_id):
                ontology = ontology_template(template_id)
                before = copy.deepcopy(ontology)
                profiled = generation_ontology(ontology)
                self.assertEqual(ontology, before)
                self.assertEqual(profiled["version"], before["version"])
                self.assertEqual(extraction_schema(profiled), extraction_schema(before))
                self.assertEqual({key: value for key, value in profiled.items()
                                  if key != "extraction_guidance"}, before)
                self.assertNotEqual(digest(profiled), digest(before))

    def test_profile_is_idempotent_and_copies_nested_records(self):
        ontology = ontology_template("tapa")
        once = generation_ontology(ontology)
        twice = generation_ontology(once)
        self.assertEqual(twice, once)
        twice["extraction_guidance"]["instructions"].append("local mutation")
        twice["predicates"]["assigned_type_by_source"]["object_types"].append("Concept")
        self.assertNotEqual(twice, once)
        self.assertEqual(once["predicates"], ontology["predicates"])
        self.assertNotIn("local mutation", generation_ontology(ontology)["extraction_guidance"]["instructions"])

    def test_manual_configuration_is_unchanged_after_validation(self):
        config = {"ontology": ontology_template("general"), "research_topic": "language contact",
                  "processing": {"mode": "manual", "max_auto_batches": 2},
                  "augmentation": {"enabled": False}}
        before = copy.deepcopy(config)
        prepared = prepare_generation_config(config)
        self.assertEqual(prepared, validate_config(config))
        self.assertEqual(config, before)
        self.assertNotIn("extraction_guidance", prepared["ontology"])
        prepared["ontology"]["entity_types"]["Concept"] = "changed locally"
        self.assertEqual(config, before)

    def test_assisted_config_preserves_topic_budgets_and_collection_policy(self):
        config = {"ontology": ontology_template("tapa"), "research_topic": "source-specific beater forms",
                  "processing": {"mode": "ollama", "batch_chars": 6000, "max_auto_batches": 4},
                  "augmentation": {"enabled": True, "max_pages": 7, "window_size": 3,
                                   "min_avg_novel_entities": 2.5}}
        before = copy.deepcopy(config)
        prepared = prepare_generation_config(config)
        validated = validate_config(config)
        self.assertEqual(config, before)
        for key in ("processing", "augmentation", "research_topic"):
            self.assertEqual(prepared[key], validated[key])
        self.assertEqual(prepared["ontology"]["extraction_guidance"]["profile_version"], GENERATION_PROFILE_VERSION)
        self.assertEqual(prepare_generation_config(prepared), prepared)
        self.assertEqual(validate_config(prepared), prepared)

    def test_invalid_configuration_and_predicate_constraints_still_fail(self):
        for config in ({"processing": {"mode": "unknown"}},
                       {"processing": {"mode": "ollama", "max_auto_batches": 101}},
                       {"augmentation": {"delay": 0}}, {"unknown_option": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                prepare_generation_config(config)
        ontology = ontology_template("general")
        ontology["predicates"]["related_to"]["object_types"] = ["UndeclaredType"]
        with self.assertRaises(ValueError):
            generation_ontology(ontology)
        for invalid in (None, [], {}, {"entity_types": {}, "predicates": {}}):
            with self.subTest(ontology=invalid), self.assertRaises(ValueError):
                generation_ontology(invalid)

    def test_guidance_reaches_existing_prompt_without_changing_output_fields(self):
        import hashlib
        text = "People beat bark with a wooden tool."
        units = [{"source_id": "s1", "source_version_id": "s1:v1", "unit_id": "u1",
                  "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "locator": {"page": 1}}]
        ontology = generation_ontology(ontology_template("general"))
        prompt = extraction_prompt(units, ontology)
        self.assertIn(GENERATION_PROFILE_VERSION, prompt)
        self.assertIn("Do not infer missing relationships to meet a graph-density target", prompt)
        self.assertIn("co-occurrence and a shared page alone do not establish a relationship", prompt)
        self.assertIn("Retain literal values for counts", prompt)
        schema = extraction_schema(ontology)
        self.assertEqual(set(schema["properties"]), {"entities", "assertions"})
        self.assertFalse(schema["additionalProperties"])

    def test_profile_is_pinned_in_new_inputs_and_existing_run_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "source.txt"
            document.write_text("People beat bark with a wooden tool.", encoding="utf-8")
            store = RunStore(root / "runs", "test-workspace")
            documents = [{"path": document, "title": "Source", "locator": "test:source"}]
            original_config = validate_config({"processing": {"mode": "ollama"}})
            original = store.create(documents, original_config, "researcher", "original")
            original_path = store.path(original["run_id"]) / "inputs.json"
            original_bytes = original_path.read_bytes()
            original_inputs = read_json(original_path)
            prepared = prepare_generation_config(original_inputs["config"])
            updated = store.create(documents, prepared, "researcher", "profiled")
            updated_inputs = read_json(store.path(updated["run_id"]) / "inputs.json")
            self.assertNotEqual(original["run_id"], updated["run_id"])
            self.assertEqual(original_path.read_bytes(), original_bytes)
            self.assertNotIn("extraction_guidance", original_inputs["config"]["ontology"])
            self.assertEqual(updated_inputs["config"]["ontology"]["extraction_guidance"]["profile_version"], GENERATION_PROFILE_VERSION)
            self.assertNotIn("graph_generation_profiles.py", updated_inputs["implementation_sha256"])
            verify_implementation(original_inputs)
            verify_implementation(updated_inputs)


if __name__ == "__main__":
    unittest.main()
