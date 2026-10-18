"""Preparation checkpoints, input identity and restart behavior without providers."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.repository.processing_jobs import ProcessingScheduler


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
        self.assertEqual(replacement.get(job["job_id"])["status"], "done")

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
