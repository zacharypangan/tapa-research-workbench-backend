"""Authenticated gateway/worker round trips using only an in-process ASGI app."""
import asyncio
import base64
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
import httpx

from app.api import model_tasks as api
from app.repository.knowledge_pipeline.ontologies import ontology_template
from app.repository.knowledge_pipeline.processing import extraction_schema
from app.repository.model_tasks import ModelTaskStore


GATEWAY = "fixture-gateway-" + "g" * 32
WORKER = "fixture-worker-" + "w" * 32
ROOT = "/api/v1"


class ModelTaskApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.now = 1000.0
        self.store = ModelTaskStore(Path(self.temporary.name) / "tasks", clock=lambda: self.now)
        self.environment = patch.dict(os.environ, {"MODEL_GATEWAY_TOKEN": GATEWAY, "MODEL_WORKER_TOKEN": WORKER})
        self.environment.start()
        self.store_patch = patch.object(api, "get_model_task_store", return_value=self.store)
        self.store_patch.start()
        self.poll_patch = patch.object(api, "POLL_SECONDS", 0.001)
        self.poll_patch.start()
        application = FastAPI()
        application.include_router(api.router, prefix=ROOT)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=application, raise_app_exceptions=False),
                                        base_url="http://fixture.invalid")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.poll_patch.stop()
        self.store_patch.stop()
        self.environment.stop()
        self.temporary.cleanup()

    async def post(self, path, body, token=WORKER):
        return await self.client.post(ROOT + path, json=body, headers={"Authorization": "Bearer " + token})

    async def claim(self, model="qwen3:8b", worker="fixture-worker"):
        return await self.post("/model-workers/claim", {"worker_id": worker, "capabilities": {"models": [model]}})

    async def enqueue(self, model="qwen3:8b", operation="chat", **kwargs):
        original = {"model": model, **({"messages": [{"role": "user", "content": "Private synthetic passage"}]} if operation == "chat" else kwargs)}
        if operation == "chat":
            original.update(kwargs)
        effective, timeout, audit = api.prepare_gateway_request(operation, original)
        return self.store.enqueue(operation, effective, timeout_seconds=timeout, request_audit=audit)

    @staticmethod
    def result_payload(task, result):
        return {"task_id": task["id"], "worker_id": task["worker_id"],
                "lease_token": task["lease_token"], "request_hash": task["request_hash"], "result": result}

    async def test_tokens_are_independent_and_both_required_to_enable_service(self):
        claim = {"worker_id": "fixture-worker", "capabilities": {"models": ["qwen3:8b"]}}
        self.assertEqual((await self.post("/model-workers/claim", claim, GATEWAY)).status_code, 401)
        self.assertEqual((await self.client.get(ROOT + "/model-workers/status", headers={"Authorization": "Bearer " + WORKER})).status_code, 401)
        self.assertEqual((await self.client.post(ROOT + "/model-workers/claim", json=claim)).status_code, 401)
        for changes in ({"MODEL_GATEWAY_TOKEN": ""}, {"MODEL_WORKER_TOKEN": "short"}, {"MODEL_WORKER_TOKEN": GATEWAY}):
            with self.subTest(changes=list(changes)), patch.dict(os.environ, changes):
                self.assertEqual((await self.post("/model-workers/claim", claim)).status_code, 503)

    async def test_no_worker_returns_503_without_queueing_or_echoing_input(self):
        response = await self.post("/model-gateway/api/chat", {"model": "qwen3:8b", "messages": [{"role": "user", "content": "Private synthetic passage"}]}, GATEWAY)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("Private synthetic passage", response.text)
        self.assertEqual(self.store.status()["pending_count"], 0)

    async def test_duplicate_keys_nonfinite_and_unknown_fields_are_rejected(self):
        for raw in ('{"worker_id":"one","worker_id":"two"}', '{"worker_id":NaN}', '[]'):
            response = await self.client.post(ROOT + "/model-workers/claim", content=raw,
                headers={"Authorization": "Bearer " + WORKER, "Content-Type": "application/json"})
            self.assertEqual(response.status_code, 400)
        response = await self.post("/model-workers/claim", {"worker_id": "fixture-worker", "capabilities": {"models": ["qwen3:8b"]}, "url": "http://unexpected.invalid"})
        self.assertEqual(response.status_code, 422)

    async def test_claim_heartbeat_and_exact_acknowledgement_replay(self):
        await self.enqueue()
        task = (await self.claim()).json()["task"]
        heartbeat = await self.post("/model-workers/heartbeat", {"task_id": task["id"], "worker_id": task["worker_id"], "lease_token": task["lease_token"]})
        self.assertEqual(heartbeat.status_code, 200)
        self.assertEqual(set(heartbeat.json()), {"status", "lease_until", "deadline_at"})
        result = {"model": "qwen3:8b", "done": True, "message": {"role": "assistant", "content": "Synthetic answer"}}
        payload = self.result_payload(task, result)
        self.assertEqual((await self.post("/model-workers/result", payload)).json(), {"status": "succeeded"})
        self.assertIsNone(self.store.get(task["id"])["body"])
        self.assertEqual((await self.post("/model-workers/result", payload)).status_code, 200)
        changed = copy.deepcopy(payload)
        changed["result"]["message"]["content"] = "Changed answer"
        self.assertEqual((await self.post("/model-workers/result", changed)).status_code, 409)
        status = await self.client.get(ROOT + "/model-workers/status", headers={"Authorization": "Bearer " + GATEWAY})
        self.assertEqual(status.status_code, 200)
        self.assertNotIn("Private synthetic passage", status.text)
        self.assertNotIn("Synthetic answer", status.text)
        self.assertNotIn(task["lease_token"], status.text)

    async def test_result_hash_stale_lease_and_terminal_tasks_are_conflicts(self):
        await self.enqueue()
        task = (await self.claim()).json()["task"]
        result = {"model": "qwen3:8b", "done": True, "message": {"content": "Synthetic answer"}}
        payload = self.result_payload(task, result)
        self.assertEqual((await self.post("/model-workers/result", {**payload, "request_hash": "0" * 64})).status_code, 409)
        self.assertEqual((await self.post("/model-workers/result", {**payload, "lease_token": "wrong-token-" + "x" * 32})).status_code, 409)
        self.now += 81
        self.assertEqual((await self.post("/model-workers/result", payload)).status_code, 409)
        await self.enqueue()
        failed = (await self.claim()).json()["task"]
        self.store.fail(failed["id"], failed["worker_id"], failed["lease_token"], "worker_error")
        self.assertEqual((await self.post("/model-workers/result", self.result_payload(failed, result))).status_code, 409)

    async def test_all_worker_failure_categories_are_accepted_by_broker(self):
        for category in sorted(api.FAILURE_CATEGORIES):
            with self.subTest(category=category):
                await self.enqueue()
                task = (await self.claim()).json()["task"]
                payload = {key: value for key, value in self.result_payload(task, {}).items() if key not in {"result", "request_hash"}}
                response = await self.post("/model-workers/failure", {**payload, "category": category})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "failed")

    async def test_gateway_embedding_roundtrip_preserves_input_hash_and_scrubs_payload(self):
        model = "qwen3-embedding:0.6b"
        await self.claim(model)
        original = {"model": model, "input": ["First private sentence", "Second private sentence"], "dimensions": 2}
        pending = asyncio.create_task(self.post("/model-gateway/api/embed", original, GATEWAY))
        try:
            task = None
            for _ in range(100):
                task = (await self.claim(model)).json()["task"]
                if task:
                    break
                await asyncio.sleep(0.005)
            self.assertIsNotNone(task)
            self.assertFalse(task["body"]["truncate"])
            result = {"model": model, "embeddings": [[1.0, 0.0], [0.0, 1.0]]}
            payload = self.result_payload(task, result)
            self.assertEqual((await self.post("/model-workers/result", payload)).status_code, 200)
            self.assertEqual((await self.post("/model-workers/result", payload)).status_code, 200)
            response = await asyncio.wait_for(pending, 2)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["embeddings"], result["embeddings"])
            audit = response.json()["workbench_model_audit"]
            self.assertEqual(audit["effective_request_sha256"], task["request_hash"])
            self.assertEqual(audit["requested_request_sha256"], api._digest("embed", original))
            self.assertIsNone(self.store.get(task["id"])["body"])
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    async def test_bad_vectors_truncation_and_wrong_model_are_rejected(self):
        await self.enqueue(model="qwen3-embedding:0.6b", operation="embed", input=["one", "two"], dimensions=2)
        task = (await self.claim("qwen3-embedding:0.6b")).json()["task"]
        for vectors in ([[1, 0]], [[1, 0], [1]], [[0, 0], [1, 0]], [[True, 0], [1, 0]], [[10**1000, 0], [1, 0]]):
            with self.subTest(vector_count=len(vectors)):
                response = await self.post("/model-workers/result", self.result_payload(task, {"model": "qwen3-embedding:0.6b", "embeddings": vectors}))
                self.assertEqual(response.status_code, 422)
        await self.enqueue()
        chat = (await self.claim(worker="fixture-chat-worker")).json()["task"]
        for result in ({"model": "qwen3:8b", "done": True, "done_reason": "length", "message": {"content": "Truncated"}},
                       {"model": "other", "done": True, "message": {"content": "Wrong model"}}):
            self.assertEqual((await self.post("/model-workers/result", self.result_payload(chat, result))).status_code, 422)

    async def test_malformed_typed_inputs_produce_422_not_server_errors(self):
        for original in ({"model": [], "messages": []},
                         {"model": "qwen3:8b", "messages": [{"role": "user", "content": "Fixture"}], "options": {"num_ctx": 10**1000}}):
            self.assertEqual((await self.post("/model-gateway/api/chat", original, GATEWAY)).status_code, 422)
        response = await self.post("/model-workers/failure", {"category": []})
        self.assertEqual(response.status_code, 422)

    async def test_only_instruct_vision_model_and_approved_operations_are_accepted(self):
        image = base64.b64encode(b"\x89PNG\r\n\x1a\nfixture").decode()
        original = {"model": "qwen3-vl:8b-instruct", "messages": [{"role": "user", "content": "Describe", "images": [image]}]}
        effective, timeout, _ = api.prepare_gateway_request("chat", original)
        self.assertEqual(timeout, 50)
        self.assertFalse(effective["think"])
        for model in ("qwen3-vl:8b", "qwen3:8b", "unknown"):
            with self.subTest(model=model):
                response = await self.post("/model-gateway/api/chat", {**original, "model": model}, GATEWAY)
                self.assertEqual(response.status_code, 422)
        self.assertEqual((await self.post("/model-gateway/api/pull", {"model": "qwen3:8b"}, GATEWAY)).status_code, 404)

    async def test_bounded_graph_policy_changes_only_effective_request_with_audit(self):
        original = {"model": "qwen3:8b", "messages": [{"role": "user", "content": "Quoted source evidence"}],
                    "format": extraction_schema(ontology_template("general")),
                    "options": {"num_ctx": 65536, "num_predict": 8192}, "think": True}
        before = copy.deepcopy(original)
        effective, timeout, audit = api.prepare_gateway_request("chat", original)
        self.assertEqual(original, before)
        self.assertEqual(timeout, 80)
        self.assertEqual(effective["options"]["num_ctx"], 16384)
        self.assertEqual(effective["options"]["num_predict"], 4096)
        self.assertFalse(effective["think"])
        self.assertEqual(effective["keep_alive"], 0)
        for key in ("entities", "assertions"):
            definition = effective["format"]["properties"][key]
            self.assertEqual(definition["maxItems"], 6)
            evidence = definition["items"]["properties"]["evidence"]
            self.assertEqual(evidence["maxItems"], 2)
            self.assertEqual(set(evidence["items"]["required"]), {"unit_id", "span_id"})
        self.assertEqual(audit["requested_request_sha256"], api._digest("chat", original))
        self.assertEqual(audit["effective_request_sha256"], api._digest("chat", effective))
        self.assertNotEqual(audit["requested_request_sha256"], audit["effective_request_sha256"])


if __name__ == "__main__":
    unittest.main()
