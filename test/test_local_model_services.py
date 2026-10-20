"""Conformance harness failures must not become false readiness claims."""

import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("model_check", Path(__file__).parents[1] / "scripts/check_local_model_services.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def fixture(path, payload):
    if path == "embed":
        return {"embeddings": [[1.0, 0], [0, 1.0]]}
    if "format" in payload:
        text = json.dumps({"tool": "stone beater", "product": "barkcloth", "quote": "Researchers used a stone beater to make barkcloth."})
    elif payload["messages"][0].get("images"):
        text = "red"
    elif "Translate" in payload["messages"][0]["content"]:
        text = "This stone beater is used to make barkcloth."
    else:
        text = "workbench-ready"
    return {"message": {"content": text}, "done": True}


class LocalServiceTests(unittest.TestCase):
    def test_all_synthetic_fixtures_pass_without_quality_claim(self):
        results = check.run_checks(fixture, "chat", "embed", "vision")
        self.assertTrue(all(result["ready"] for result in results.values()))
        self.assertFalse(results["structured_extraction"]["full_workbench_validation"])
        self.assertTrue(results["translation"]["needs_human_review"])

    def test_unavailable_services_are_independent_and_errors_are_sanitized(self):
        def unavailable(path, payload):
            if path == "embed":
                raise RuntimeError("secret provider body or URL")
            return fixture(path, payload)
        result = check.run_checks(unavailable, "chat", "embed", "vision")
        self.assertTrue(result["generation"]["ready"])
        self.assertFalse(result["embeddings"]["ready"])
        self.assertNotIn("secret", json.dumps(result))

    def test_nan_vectors_and_unquoted_extraction_are_rejected(self):
        def invalid(path, payload):
            if path == "embed":
                return {"embeddings": [[float("nan")], [1]]}
            if "format" in payload:
                return {"message": {"content": json.dumps({"tool": "stone beater", "product": "barkcloth", "quote": "Invented evidence"})}}
            return fixture(path, payload)
        result = check.run_checks(invalid, "chat", "embed", "vision")
        self.assertFalse(result["embeddings"]["ready"])
        self.assertFalse(result["structured_extraction"]["ready"])

    def test_missing_models_are_not_marked_ready(self):
        self.assertTrue(all(item["state"] == "not_configured" for item in check.run_checks(fixture, None, None, None).values()))

    def test_contradictory_color_response_does_not_pass_vision(self):
        def wrong_color(path, payload):
            if payload.get("messages", [{}])[0].get("images"):
                return {"message": {"content": "blue, not red"}}
            return fixture(path, payload)
        self.assertFalse(check.run_checks(wrong_color, "chat", "embed", "vision")["vision"]["ready"])
