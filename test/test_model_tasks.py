"""Private broker leases, bounded storage and replay; no providers or credentials."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.repository.model_tasks import (
    LEASE_SECONDS, MAX_ATTEMPTS, RETENTION_SECONDS,
    ModelTaskCapacity, ModelTaskConflict, ModelTaskStore, task_request_hash,
)


MODEL = "qwen3:8b"
CAPABILITIES = {"models": [MODEL]}


def request_body():
    return {"model": MODEL, "messages": [{"role": "user", "content": "Private synthetic evidence"}],
            "stream": False}


def result_body():
    return {"model": MODEL, "done": True, "message": {"role": "assistant", "content": "Fixture answer"}}


class ModelTaskStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.now = 1000.0
        self.store = ModelTaskStore(Path(self.temporary.name) / "tasks", clock=lambda: self.now)

    def enqueue(self, **kwargs):
        return self.store.enqueue("chat", request_body(), **kwargs)

    def test_hash_binds_operation_and_exact_request(self):
        body = request_body()
        task = self.enqueue()
        self.assertEqual(task["request_hash"], task_request_hash("chat", body))
        self.assertNotEqual(task["request_hash"], task_request_hash("generate", body))
        body["messages"][0]["content"] += " changed"
        self.assertNotEqual(task["request_hash"], task_request_hash("chat", body))
        self.assertEqual(self.store.get(task["id"])["body"], request_body())

    def test_claim_is_exclusive_across_concurrent_workers_and_capabilities(self):
        self.enqueue()
        self.assertIsNone(self.store.claim("other-model", {"models": ["qwen3-embedding:0.6b"]}))
        with ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(executor.map(lambda worker: self.store.claim(worker, CAPABILITIES), ["one", "two"]))
        claimed = [task for task in claims if task is not None]
        self.assertEqual(len(claimed), 1)
        task = claimed[0]
        self.assertEqual(self.store.claim(task["worker_id"], CAPABILITIES)["lease_token"], task["lease_token"])

    def test_expired_lease_reassigns_and_rejects_old_worker(self):
        self.enqueue()
        first = self.store.claim("first", CAPABILITIES)
        self.now += LEASE_SECONDS + 1
        second = self.store.claim("second", CAPABILITIES)
        self.assertEqual(first["id"], second["id"])
        self.assertNotEqual(first["lease_token"], second["lease_token"])
        with self.assertRaises(ModelTaskConflict):
            self.store.complete(first["id"], "first", first["lease_token"], result_body())
        self.assertEqual(self.store.complete(second["id"], "second", second["lease_token"], result_body())["status"], "succeeded")

    def test_heartbeat_is_owned_and_cannot_extend_absolute_deadline(self):
        task = self.enqueue(timeout_seconds=35)
        claim = self.store.claim("worker", CAPABILITIES)
        self.now += 15
        with self.assertRaises(ModelTaskConflict):
            self.store.heartbeat(task["id"], "other", claim["lease_token"])
        renewed = self.store.heartbeat(task["id"], "worker", claim["lease_token"])
        self.assertEqual(renewed["lease_until"], task["deadline_at"])
        self.now = task["deadline_at"]
        with self.assertRaises(ModelTaskConflict):
            self.store.complete(task["id"], "worker", claim["lease_token"], result_body())
        expired = self.store.get(task["id"])
        self.assertEqual(expired["status"], "expired")
        self.assertIsNone(expired["body"])

    def test_completion_requires_hash_lease_and_exact_replay(self):
        self.enqueue()
        claim = self.store.claim("worker", CAPABILITIES)
        for worker, lease, request_hash in (("other", claim["lease_token"], claim["request_hash"]),
                                            ("worker", "invalid-lease", claim["request_hash"]),
                                            ("worker", claim["lease_token"], "0" * 64)):
            with self.subTest(worker=worker, lease=lease), self.assertRaises(ModelTaskConflict):
                self.store.complete(claim["id"], worker, lease, result_body(), request_hash=request_hash)
        result = result_body()
        completed = self.store.complete(claim["id"], "worker", claim["lease_token"], result,
                                        request_hash=claim["request_hash"])
        self.assertIsNone(completed["body"])
        self.now += 100
        self.assertEqual(self.store.complete(claim["id"], "worker", claim["lease_token"], result,
                                             request_hash=claim["request_hash"]), completed)
        changed = result_body()
        changed["message"]["content"] = "Different answer"
        with self.assertRaises(ModelTaskConflict):
            self.store.complete(claim["id"], "worker", claim["lease_token"], changed)

    def test_failure_backoff_and_attempt_ceiling(self):
        self.enqueue()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            claim = self.store.claim("worker", CAPABILITIES)
            self.assertEqual(claim["attempts"], attempt)
            failed = self.store.fail(claim["id"], "worker", claim["lease_token"], "resource_busy", retryable=True)
            if attempt < MAX_ATTEMPTS:
                self.assertEqual(failed["status"], "queued")
                self.assertIsNone(self.store.claim("worker", CAPABILITIES))
                self.now += 2 ** attempt
            else:
                self.assertEqual(failed["status"], "failed")
                self.assertFalse(failed["error"]["retryable"])
                self.assertIsNone(failed["body"])

    def test_expired_attempt_ceiling_is_terminal(self):
        task = self.enqueue()
        for attempt in range(MAX_ATTEMPTS):
            self.assertEqual(self.store.claim("worker", CAPABILITIES)["attempts"], attempt + 1)
            self.now += LEASE_SECONDS + 1
        failed = self.store.get(task["id"])
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"]["category"], "attempts_exhausted")

    def test_embedding_rejects_missing_zero_nonfinite_and_mixed_dimension_vectors(self):
        task = self.store.enqueue("embed", {"model": "qwen3-embedding:0.6b", "input": ["one", "two"]})
        claim = self.store.claim("worker", {"models": ["qwen3-embedding:0.6b"]})
        for vectors in ([[1]], [[0, 0], [1, 0]], [[1, 0], [1]], [[float("nan")], [1]], [[True], [1]]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                self.store.complete(task["id"], "worker", claim["lease_token"], {"embeddings": vectors})
        self.assertEqual(self.store.complete(task["id"], "worker", claim["lease_token"],
                                             {"embeddings": [[1, 0], [0, 1]]})["status"], "succeeded")

    def test_capacity_reserve_private_files_and_status_without_payloads(self):
        task = self.enqueue()
        self.store.claim("worker", CAPABILITIES)
        status = json.dumps(self.store.status())
        self.assertNotIn("Private synthetic evidence", status)
        self.assertNotIn("lease_token", status)
        self.assertEqual(os.stat(self.store.root).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(self.store.root / (task["id"] + ".json")).st_mode & 0o777, 0o600)
        with patch("app.repository.model_tasks.MAX_PENDING", 1), self.assertRaises(ModelTaskCapacity):
            self.enqueue()
        with patch("app.repository.model_tasks.shutil.disk_usage", return_value=SimpleNamespace(free=1)), self.assertRaises(ModelTaskCapacity):
            self.enqueue()

    def test_retention_removes_finished_payload_and_audit_hash_is_checked(self):
        with self.assertRaises(ValueError):
            self.enqueue(request_audit={"effective_request_sha256": "0" * 64})
        task = self.enqueue()
        claim = self.store.claim("worker", CAPABILITIES)
        self.store.complete(task["id"], "worker", claim["lease_token"], result_body())
        self.now += RETENTION_SECONDS + 1
        self.store.status()
        with self.assertRaises(FileNotFoundError):
            self.store.get(task["id"])


if __name__ == "__main__":
    unittest.main()
