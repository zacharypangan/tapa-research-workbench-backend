"""Preparation API status/retry semantics without providers or production data."""

import tempfile
from contextlib import ExitStack
from unittest import TestCase
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import processing
from app.repository.processing_jobs import ProcessingScheduler


class ProcessingApiTests(TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.scheduler = ProcessingScheduler(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(processing, "get_processing_scheduler", return_value=self.scheduler))
        self.stack.enter_context(patch("app.repository.processing_jobs.material_inputs", return_value={
            "material_id": "fixture", "files": [], "source_url": None, "language": "en"}))
        self.stack.enter_context(patch("app.api.repository.get_connection", return_value=MagicMock()))
        self.stack.enter_context(patch("app.api.repository.ensure_material", return_value={"id": "fixture"}))
        self.stack.enter_context(patch("app.api.repository.embedding_coverage", return_value={"embedded_segment_count": 1}))
        app = FastAPI()
        app.include_router(processing.router, prefix="/api/v1")
        self.client = self.stack.enter_context(TestClient(app))

    def test_material_status_returns_current_replacement_and_keeps_history(self):
        original = self.scheduler.enqueue("fixture", ("text_index",))
        newer = self.scheduler.enqueue("fixture", ("text_index",), force=True)
        self.scheduler.update(original["job_id"], status="done")
        self.scheduler.update(newer["job_id"], status="superseded", replacement_job_id=original["job_id"])
        result = self.client.get("/api/v1/repository/materials/fixture/processing")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["job"]["job_id"], original["job_id"])
        historical = self.client.get("/api/v1/repository/processing-jobs/" + newer["job_id"])
        self.assertEqual(historical.json()["status"], "superseded")

    def test_invalid_replacement_reports_actionable_error_without_paths(self):
        job = self.scheduler.enqueue("fixture", ("text_index",))
        self.scheduler.update(job["job_id"], status="superseded", replacement_job_id="../../secret")
        result = self.client.get("/api/v1/repository/materials/fixture/processing")
        self.assertEqual(result.status_code, 422)
        self.assertEqual(result.json()["detail"], "Preparation replacement could not be resolved")

    def test_explicit_retry_follows_replacement_and_resets_permanent_failure(self):
        original = self.scheduler.enqueue("fixture", ("text_index",))
        newer = self.scheduler.enqueue("fixture", ("text_index",), force=True)
        self.scheduler.update(original["job_id"], status="error", stage_errors={"text_index": {"retryable": False}})
        self.scheduler.update(newer["job_id"], status="superseded", replacement_job_id=original["job_id"])
        result = self.client.post("/api/v1/repository/processing-jobs/" + newer["job_id"] + "/retry")
        self.assertEqual(result.status_code, 202)
        self.assertEqual(result.json()["job_id"], original["job_id"])
        self.assertEqual(result.json()["stage_errors"], {})
        self.assertEqual(result.json()["status"], "queued")
