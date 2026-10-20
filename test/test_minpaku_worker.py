"""Worker admission/security and lease tests; no credentials or network required."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest


SPEC = importlib.util.spec_from_file_location("minpaku_worker", Path(__file__).parents[1] / "scripts/minpaku_pilot/worker.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def task():
    body = {"model": "qwen3:8b", "messages": [{"role": "user", "content": "Synthetic fixture"}],
            "stream": False, "think": False, "keep_alive": 0,
            "options": {"num_ctx": 16384, "num_predict": 32, "temperature": 0}}
    return {"id": "test-task", "operation": "chat", "body": body,
            "request_hash": MODULE.request_hash("chat", body), "worker_id": "fixture-worker",
            "lease_token": "fixture-lease", "deadline_at": time.time() + 60, "lease_until": time.time() + 20}


class WorkerTests(unittest.TestCase):
    def config(self):
        return {"backend_base_url": "https://" + MODULE.STAGING_HOST + "/api/v1",
                "worker_id": "fixture-worker", "token": "fixture-" + "x" * 32}

    def test_only_private_owned_staging_config_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "worker.json"
            path.write_text(json.dumps(self.config()))
            path.chmod(0o600)
            self.assertEqual(MODULE.load_config(path, root)["worker_id"], "fixture-worker")
            for url in ("http://" + MODULE.STAGING_HOST, "https://production.example/api/v1",
                        "https://user@" + MODULE.STAGING_HOST, "https://" + MODULE.STAGING_HOST + "/api/v1?token=x"):
                path.write_text(json.dumps({**self.config(), "backend_base_url": url}))
                with self.subTest(url=url), self.assertRaises(ValueError):
                    MODULE.load_config(path, root)
            path.write_text(json.dumps(self.config()))
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                MODULE.load_config(path, root)

    def test_only_fixed_loopback_runtime_origin(self):
        for origin in ("http://example.test", "https://127.0.0.1:11437", "http://127.0.0.1:11437/api/pull",
                       "http://user@127.0.0.1:11437"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                MODULE.Worker(self.config(), origin)

    def test_hash_includes_operation_and_body_exactly(self):
        current = task()
        MODULE.validate_task(current)
        mutated = copy.deepcopy(current)
        mutated["body"]["messages"][0]["content"] = "Changed"
        with self.assertRaises(ValueError):
            MODULE.validate_task(mutated)
        self.assertNotEqual(MODULE.request_hash("chat", current["body"]), MODULE.request_hash("generate", current["body"]))

    def test_rejects_unapproved_model_endpoint_and_unbounded_options(self):
        cases = [lambda item: item.update(operation="pull"),
                 lambda item: item["body"].update(model="other-model"),
                 lambda item: item["body"].update(stream=True),
                 lambda item: item["body"].update(think=True),
                 lambda item: item["body"].update(keep_alive="24h"),
                 lambda item: item["body"]["options"].update(num_ctx=32768),
                 lambda item: item["body"]["options"].update(num_predict=-1),
                 lambda item: item["body"]["options"].update(num_gpu=0),
                 lambda item: item.update(deadline_at=time.time() - 1)]
        for mutate in cases:
            current = task()
            mutate(current)
            current["request_hash"] = MODULE.request_hash(current["operation"], current["body"])
            with self.subTest(task=current), self.assertRaises(ValueError):
                MODULE.validate_task(current)

    def worker(self):
        worker = MODULE.Worker(self.config(), "http://127.0.0.1:11437")
        worker.capabilities = {"models": list(MODULE.APPROVED_MODELS),
                               "digests": {name: "a" * 64 for name in MODULE.APPROVED_MODELS}}
        worker.heartbeat_seconds = 0.005
        return worker

    def test_heartbeat_renews_lease_and_exact_result_hash_is_returned(self):
        worker, calls = self.worker(), []

        def broker(operation, payload):
            calls.append((operation, payload))
            return {"lease_until": time.time() + 20, "status": "running"}

        def local(url, payload=None, **kwargs):
            time.sleep(0.025)
            return {"model": payload["model"], "done": True, "message": {"content": "OK"}}

        worker.broker, worker.http = broker, local
        current = task()
        self.assertEqual(worker.process_task(current), "succeeded")
        self.assertTrue(any(operation == "heartbeat" for operation, payload in calls))
        result = next(payload for operation, payload in calls if operation == "result")
        self.assertEqual(result["request_hash"], current["request_hash"])
        self.assertEqual(result["audit"], {"model_digest": "a" * 64})

    def test_lost_lease_discards_result_without_submission(self):
        worker, calls = self.worker(), []

        def broker(operation, payload):
            calls.append(operation)
            if operation == "heartbeat":
                raise MODULE.HTTPError("fixture", 409, "Lost lease", {}, None)
            return {}

        def local(url, payload=None, **kwargs):
            time.sleep(0.025)
            return {"model": payload["model"], "done": True, "message": {"content": "OK"}}

        worker.broker, worker.http = broker, local
        self.assertEqual(worker.process_task(task()), "lease_lost")
        self.assertNotIn("result", calls)
        self.assertNotIn("failure", calls)

    def test_embedding_legacy_model_attribution_is_explicit(self):
        worker, calls = self.worker(), []
        worker.broker = lambda operation, payload: calls.append((operation, payload)) or {}
        worker.http = lambda *args, **kwargs: {"embedding": [0.1, 0.2]}
        current = task()
        current["operation"] = "embeddings"
        current["body"] = {"model": MODULE.APPROVED_MODELS[2], "prompt": "Synthetic fixture", "keep_alive": 0}
        current["request_hash"] = MODULE.request_hash(current["operation"], current["body"])
        self.assertEqual(worker.process_task(current), "succeeded")
        result = next(payload for operation, payload in calls if operation == "result")
        self.assertTrue(result["audit"]["response_model_inferred"])


if __name__ == "__main__":
    unittest.main()
