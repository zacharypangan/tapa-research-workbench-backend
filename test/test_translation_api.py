from contextlib import ExitStack
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import translations
from app.repository.translations import TranslationScheduler


class TranslationApiTests(TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.scheduler = TranslationScheduler(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(translations, "get_translation_scheduler", return_value=self.scheduler))
        self.inputs = {"material_id": "fixture", "files": [{"file_id": "paper", "sha256": "a"*64, "filename": "paper.pdf"}], "language": "zh"}
        self.stack.enter_context(patch("app.repository.translations.material_inputs", return_value=self.inputs))
        self.stack.enter_context(patch.object(translations, "material_inputs", return_value=self.inputs))
        self.stack.enter_context(patch("app.repository.translations.configured_model", return_value="fixture-model"))
        self.stack.enter_context(patch("app.api.repository.get_connection", return_value=MagicMock()))
        self.stack.enter_context(patch("app.api.repository.ensure_material", return_value={"id": "fixture"}))
        app = FastAPI()
        app.include_router(translations.router, prefix="/api/v1")
        self.client = self.stack.enter_context(TestClient(app))

    def test_create_is_idempotent_and_queued_artifact_is_available(self):
        route = "/api/v1/repository/materials/fixture/translations"
        first = self.client.post(route, json={})
        second = self.client.post(route, json={})
        self.assertEqual(first.status_code, 202)
        self.assertEqual(first.json()["job_id"], second.json()["job_id"])
        job_route = "/api/v1/repository/translation-jobs/" + first.json()["job_id"]
        artifact = self.client.get(job_route + "/artifact")
        self.assertEqual(artifact.status_code, 200)
        self.assertEqual(artifact.json()["units"], [])
        self.assertFalse(artifact.json()["stale"])
        download = self.client.get(job_route + "/artifact?format=text")
        self.assertIn("Machine translation", download.text)
        self.assertIn("attachment;", download.headers["content-disposition"])
        self.assertEqual(self.client.get(route).json()["jobs"][0]["status"], "queued")

    def test_input_validation_and_invalid_job_do_not_expose_paths(self):
        self.assertEqual(self.client.post("/api/v1/repository/materials/fixture/translations", json={"file_id": ""}).status_code, 422)
        response = self.client.get("/api/v1/repository/translation-jobs/not-a-uuid")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Translation job not found.")

    def test_retry_preserves_done_and_listing_marks_changed_source(self):
        job = self.scheduler.enqueue("fixture")
        self.scheduler.update(job["job_id"], status="done")
        route = "/api/v1/repository/translation-jobs/" + job["job_id"]
        self.assertEqual(self.client.post(route+"/retry").json()["status"], "done")
        changed = {**self.inputs, "files": []}
        with patch.object(translations, "material_inputs", return_value=changed):
            listing = self.client.get("/api/v1/repository/materials/fixture/translations").json()
        self.assertTrue(listing["jobs"][0]["stale"])
