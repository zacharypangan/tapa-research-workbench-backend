"""Preparation checkpoints, input identity and restart behavior without providers."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.repository.processing_jobs import ProcessingScheduler, ProcessingStageError


class PreparationJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = {"material_id": "fixture", "files": [{"file_id": "f1", "sha256": "test", "filename": "fixture.txt"}],
                       "source_url": None, "language": "en"}
        self.patcher = patch("app.repository.processing_jobs.material_inputs", return_value=self.inputs)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_enqueue_is_idempotent_and_reopens_from_disk(self):
        scheduler = ProcessingScheduler(self.root)
        first = scheduler.enqueue("fixture")
        second = ProcessingScheduler(self.root).enqueue("fixture")
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(first["input_hash"], second["input_hash"])
        self.assertNotEqual(first["job_id"], scheduler.enqueue("fixture", force=True)["job_id"])

    def test_checkpoint_survives_failure_and_new_scheduler(self):
        calls = []

        async def runner(stage, job):
            calls.append(stage)
            if stage == "text_index":
                raise RuntimeError("synthetic transient failure")
            return {"synthetic": True}

        scheduler = ProcessingScheduler(self.root, stage_runner=runner)
        job = scheduler.enqueue("fixture", ("extract", "text_index"))
        with patch("app.repository.processing_jobs.asyncio.sleep", return_value=None):
            asyncio.run(scheduler.process(job["job_id"]))
        stored = scheduler.get(job["job_id"])
        self.assertEqual(stored["status"], "queued")
        self.assertIn("extract", stored["completed_stages"])

        async def recovered(stage, job):
            calls.append(stage)
            return {"synthetic": True}

        replacement = ProcessingScheduler(self.root, stage_runner=recovered)
        asyncio.run(replacement.process(job["job_id"]))
        self.assertEqual(calls, ["extract", "text_index", "text_index"])
        final = replacement.get(job["job_id"])
        self.assertEqual(final["status"], "done")
        self.assertEqual(final["stage_errors"], {})
        self.assertIsNone(final["detail"])

    def test_failed_text_index_does_not_block_image_index(self):
        calls = []
        async def runner(stage, job):
            calls.append(stage)
            if stage == "text_index":
                raise RuntimeError("synthetic embedding outage")
            return {"synthetic": True}
        scheduler = ProcessingScheduler(self.root, stage_runner=runner)
        job = scheduler.enqueue("fixture", ("text_index", "image_index"))
        with patch("app.repository.processing_jobs.asyncio.sleep", return_value=None):
            asyncio.run(scheduler.process(job["job_id"]))
        final = scheduler.get(job["job_id"])
        self.assertEqual(calls, ["text_index", "image_index"])
        self.assertEqual(final["status"], "queued")
        self.assertIn("image_index", final["completed_stages"])
        self.assertEqual(set(final["stage_errors"]), {"text_index"})

    def test_input_change_during_failed_final_stage_supersedes_plan(self):
        changed = {**self.inputs, "source_url": "https://example.org/new"}
        scheduler = ProcessingScheduler(self.root)
        job = scheduler.enqueue("fixture", ("text_index",))
        # Keep the mock handle so the runner can simulate a real upload mid-stage.
        with patch("app.repository.processing_jobs.material_inputs", return_value=self.inputs) as inputs:
            async def mutate(stage, job):
                inputs.return_value = changed
                raise RuntimeError("obsolete input failed")
            scheduler.stage_runner = mutate
            asyncio.run(scheduler.process(job["job_id"]))
        final = scheduler.get(job["job_id"])
        self.assertEqual(final["status"], "superseded")
        replacement = scheduler.get(final["replacement_job_id"])
        self.assertEqual(replacement["stages"], ["text_index"])
        self.assertEqual(replacement["inputs"], changed)

    def test_index_pass_processes_healthy_items_after_failed_items_once(self):
        scheduler = ProcessingScheduler(self.root)
        job = scheduler.enqueue("fixture", ("text_index",))
        responses = [
            {"indexed_count": 1, "pending_count": 2, "failed_count": 1,
             "failed_ids": ["bad"], "outcomes": [{"segment_id": "bad", "status": "failed"}]},
            {"indexed_count": 1, "pending_count": 1, "failed_count": 0,
             "failed_ids": [], "outcomes": [{"segment_id": "healthy", "status": "indexed"}]},
        ]
        operation = AsyncMock(side_effect=responses)
        connection = MagicMock()
        with patch("app.api.repository.ai_embedding_configured", return_value=True), \
                patch("app.api.repository.get_connection", return_value=connection), \
                patch("app.api.repository.index_missing_segment_embeddings", operation):
            with self.assertRaisesRegex(RuntimeError, "1 item.*need retry"):
                asyncio.run(scheduler.run_stage("text_index", job))
        self.assertEqual(operation.await_count, 2)
        self.assertEqual(operation.await_args_list[1].kwargs["skip_ids"], {"bad"})
        progress = scheduler.get(job["job_id"])["stage_progress"]["text_index"]
        self.assertEqual(progress["indexed_count"], 2)
        self.assertEqual(progress["failed_ids"], ["bad"])

    def test_three_failures_stop_and_retry_is_explicit(self):
        async def failure(stage, job):
            raise RuntimeError("synthetic outage")
        scheduler = ProcessingScheduler(self.root, stage_runner=failure)
        job = scheduler.enqueue("fixture", ("text_index",))
        with patch("app.repository.processing_jobs.asyncio.sleep", return_value=None):
            for _ in range(3):
                asyncio.run(scheduler.process(job["job_id"]))
        self.assertEqual(scheduler.get(job["job_id"])["status"], "error")
        retried = scheduler.retry(job["job_id"])
        self.assertEqual(retried["status"], "queued")
        self.assertEqual(retried["attempts"], 0)

    def test_permanent_stage_failure_stops_automatic_retry_and_explicit_retry_probes(self):
        calls = []
        async def runner(stage, job):
            calls.append(stage)
            raise ProcessingStageError("Provider HTTP 404 needs configuration correction", retryable=False)
        scheduler = ProcessingScheduler(self.root, stage_runner=runner)
        job = scheduler.enqueue("fixture", ("text_index",))
        asyncio.run(scheduler.process(job["job_id"]))
        final = scheduler.get(job["job_id"])
        self.assertEqual(final["status"], "error")
        self.assertFalse(final["retryable"])
        self.assertEqual(final["attempts"], 1)
        self.assertFalse(asyncio.run(scheduler.process(job["job_id"])))
        self.assertEqual(calls, ["text_index"])
        retried = scheduler.retry(job["job_id"])
        self.assertEqual(retried["stage_errors"], {})
        asyncio.run(scheduler.process(job["job_id"]))
        self.assertEqual(calls, ["text_index", "text_index"])

    def test_permanent_stage_is_skipped_while_other_transient_stage_recovers(self):
        calls = []
        async def runner(stage, job):
            calls.append(stage)
            if stage == "text_index":
                raise ProcessingStageError("Missing embedding model", retryable=False)
            if calls.count("image_index") == 1:
                raise ProcessingStageError("Rate limited", retryable=True)
            return {"synthetic": True}
        scheduler = ProcessingScheduler(self.root, stage_runner=runner)
        job = scheduler.enqueue("fixture", ("text_index", "image_index"))
        with patch("app.repository.processing_jobs.asyncio.sleep", return_value=None):
            asyncio.run(scheduler.process(job["job_id"]))
            self.assertEqual(scheduler.get(job["job_id"])["status"], "queued")
            asyncio.run(scheduler.process(job["job_id"]))
        self.assertEqual(calls, ["text_index", "image_index", "image_index"])
        final = scheduler.get(job["job_id"])
        self.assertEqual(final["status"], "error")
        self.assertFalse(final["retryable"])
        self.assertEqual(set(final["stage_errors"]), {"text_index"})

    def test_index_permanent_outcomes_are_aggregated_without_automatic_retry(self):
        scheduler = ProcessingScheduler(self.root)
        job = scheduler.enqueue("fixture", ("text_index",))
        result = {"indexed_count": 0, "pending_count": 2, "failed_count": 2, "failed_ids": [1, 2],
                  "outcomes": [{"id": item, "status": "failed", "error_category": "provider_http_error",
                                "http_status": 404, "retryable": False} for item in [1, 2]]}
        with patch("app.api.repository.ai_embedding_configured", return_value=True), \
                patch("app.api.repository.get_connection", return_value=MagicMock()), \
                patch("app.api.repository.index_missing_segment_embeddings", AsyncMock(return_value=result)):
            with self.assertRaises(ProcessingStageError) as caught:
                asyncio.run(scheduler.run_stage("text_index", job))
        self.assertFalse(caught.exception.retryable)
        summary = scheduler.get(job["job_id"])["stage_progress"]["text_index"]["error_summary"]
        self.assertEqual(summary, [{"error_category": "provider_http_error", "http_status": 404, "retryable": False, "count": 2}])

    def test_current_status_and_retry_follow_replacement_with_older_timestamp(self):
        scheduler = ProcessingScheduler(self.root)
        original = scheduler.enqueue("fixture", ("text_index",))
        newer = scheduler.enqueue("fixture", ("text_index",), force=True)
        scheduler.update(original["job_id"], status="error")
        scheduler.update(newer["job_id"], status="superseded", replacement_job_id=original["job_id"])
        self.assertEqual(scheduler.current_for("fixture")["job_id"], original["job_id"])
        self.assertEqual(scheduler.retry(newer["job_id"])["job_id"], original["job_id"])
        scheduler.update(original["job_id"], status="superseded", replacement_job_id=newer["job_id"])
        with self.assertRaisesRegex(ValueError, "cycle"):
            scheduler.current_for("fixture")

    def test_retry_group_requeues_failed_children(self):
        scheduler = ProcessingScheduler(self.root)
        with patch("app.repository.processing_jobs.material_inputs", side_effect=lambda material: {**self.inputs, "material_id": material}):
            group = scheduler.enqueue_group(["fixture", "other"], ("text_index",))
        child, completed = group["child_job_ids"]
        checkpoint = {"text_index": {"finished_at": "saved", "result": {"synthetic": True}}}
        scheduler.update(completed, status="done", completed_stages=checkpoint)
        scheduler.update(child, status="error", stage_errors={"text_index": {"retryable": False}})
        asyncio.run(scheduler.process(group["job_id"]))
        self.assertEqual(scheduler.get(group["job_id"])["status"], "error")
        scheduler.retry(group["job_id"])
        self.assertEqual(scheduler.get(child)["status"], "queued")
        self.assertEqual(scheduler.get(child)["stage_errors"], {})
        self.assertEqual(scheduler.get(completed)["status"], "done")
        self.assertEqual(scheduler.get(completed)["completed_stages"], checkpoint)

    def test_new_upload_supersedes_old_plan(self):
        calls = []
        async def runner(stage, job):
            calls.append(stage)
            return {}
        scheduler = ProcessingScheduler(self.root, stage_runner=runner)
        job = scheduler.enqueue("fixture")
        changed = {**self.inputs, "files": self.inputs["files"] + [{"file_id": "f2", "sha256": "new", "filename": "new.txt"}]}
        with patch("app.repository.processing_jobs.material_inputs", return_value=changed):
            asyncio.run(scheduler.process(job["job_id"]))
        current = scheduler.get(job["job_id"])
        self.assertEqual(current["status"], "superseded")
        self.assertNotEqual(current["replacement_job_id"], current["job_id"])
        self.assertEqual(calls, [])

    def test_material_lock_prevents_duplicate_processing(self):
        async def scenario():
            entered, finish = asyncio.Event(), asyncio.Event()
            async def runner(stage, job):
                entered.set()
                await finish.wait()
                return {}
            scheduler = ProcessingScheduler(self.root, stage_runner=runner)
            job = scheduler.enqueue("fixture", ("extract",))
            first = asyncio.create_task(scheduler.process(job["job_id"]))
            await entered.wait()
            self.assertFalse(await ProcessingScheduler(self.root, stage_runner=runner).process(job["job_id"]))
            finish.set()
            self.assertTrue(await first)
        asyncio.run(scenario())

    def test_cancellation_keeps_job_recoverable(self):
        async def scenario():
            entered, finish = asyncio.Event(), asyncio.Event()
            async def runner(stage, job):
                entered.set()
                await finish.wait()
                return {}
            scheduler = ProcessingScheduler(self.root, stage_runner=runner)
            job = scheduler.enqueue("fixture", ("extract",))
            task = asyncio.create_task(scheduler.process(job["job_id"]))
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(scheduler.get(job["job_id"])["status"], "queued")
        asyncio.run(scenario())

    def test_invalid_job_id_cannot_escape_storage(self):
        with self.assertRaises(ValueError):
            ProcessingScheduler(self.root).get("../../repository.sqlite")


if __name__ == "__main__":
    unittest.main()
