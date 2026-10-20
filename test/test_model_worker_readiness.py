"""Metadata-only bridge readiness fails closed without inference or redirects."""
import copy
import unittest
from unittest.mock import patch

import httpx

from app.repository.model_tasks import READINESS_VERSION
from app.repository.model_worker_readiness import STAGING_GATEWAY, bridge_readiness


def health():
    return {"version": READINESS_VERSION, "checked_at": 1000,
            "worker_max_age_seconds": 35, "success_max_age_seconds": 300,
            "queued_count": 2, "running_count": 1, "pending_count": 3,
            "models": {"qwen3:8b": {"ready": True, "worker_fresh": True,
                "last_seen_at": 999, "last_success_at": 990, "last_success_operation": "chat"}}}


class BridgeReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def probe(self, payload=None, *, status=200, handler=None, base=STAGING_GATEWAY, model="qwen3:8b"):
        calls = []
        def respond(request):
            calls.append(request)
            if handler:
                return handler(request)
            return httpx.Response(status, json=payload if payload is not None else health())
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch("app.repository.model_worker_readiness.httpx.AsyncClient", return_value=client) as constructor, patch(
                "app.repository.model_worker_readiness.time.time", return_value=1000):
            result = await bridge_readiness(base, {"Authorization": "Bearer synthetic-fixture"}, model)
        return result, calls, constructor.call_args

    async def test_recent_execution_with_busy_queue_is_ready_without_model_call(self):
        result, calls, constructor = await self.probe()
        self.assertTrue(result[0])
        self.assertIn("2 waiting, 1 running", result[1])
        self.assertIn("10 seconds ago", result[1])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].method, "GET")
        self.assertEqual(str(calls[0].url), STAGING_GATEWAY + "/health")
        self.assertEqual(calls[0].headers["authorization"], "Bearer synthetic-fixture")
        self.assertEqual(constructor.kwargs, {"timeout": 5.0, "follow_redirects": False})

    async def test_other_origins_production_and_ambiguous_urls_keep_existing_probes(self):
        for base in ("http://localhost:11434", "https://ollama.com", STAGING_GATEWAY.replace("graph-staging", "production"),
                     STAGING_GATEWAY + "?token=x", STAGING_GATEWAY + "#fragment", STAGING_GATEWAY.replace("https://", "http://"),
                     STAGING_GATEWAY.replace("https://", "https://user@")):
            with self.subTest(base=base), patch("app.repository.model_worker_readiness.httpx.AsyncClient") as client:
                self.assertIsNone(await bridge_readiness(base, {}, "qwen3:8b"))
                client.assert_not_called()
        result, calls, _ = await self.probe(base=STAGING_GATEWAY + "/api/")
        self.assertTrue(result[0])
        self.assertEqual(str(calls[0].url), STAGING_GATEWAY + "/health")

    async def test_heartbeat_alone_and_disconnected_worker_are_not_ready(self):
        payload = health()
        row = payload["models"]["qwen3:8b"]
        row.update(ready=False, last_success_at=None, last_success_operation=None)
        result, _, _ = await self.probe(payload)
        self.assertFalse(result[0])
        self.assertIn("execution", result[1])
        row.update(worker_fresh=False, last_seen_at=950, last_success_at=990, last_success_operation="chat")
        result, _, _ = await self.probe(payload)
        self.assertFalse(result[0])
        self.assertIn("disconnected", result[1])

    async def test_stale_wrong_operation_or_future_proof_cannot_pass_ready_flag(self):
        for changes in ({"last_success_at": 699}, {"last_success_at": 1001},
                        {"last_success_operation": "embed"}, {"last_seen_at": 964},
                        {"worker_fresh": 1}, {"ready": "true"}):
            payload = health()
            payload["models"]["qwen3:8b"].update(changes)
            with self.subTest(changes=changes):
                self.assertFalse((await self.probe(payload))[0][0])

    async def test_stale_health_malformed_metadata_and_http_errors_fail_closed(self):
        variants = [None, [], {}, {**health(), "checked_at": 989}, {**health(), "success_max_age_seconds": 301},
                    {**health(), "queued_count": True}, {**health(), "models": {}}]
        for payload in variants[1:]:
            with self.subTest(payload_type=type(payload).__name__):
                self.assertFalse((await self.probe(payload))[0][0])
        for status in (302, 401, 503):
            result, calls, _ = await self.probe({"detail": "sensitive provider message"}, status=status)
            self.assertFalse(result[0])
            self.assertNotIn("sensitive", result[1])
            self.assertEqual(len(calls), 1)
        def timeout(request):
            raise httpx.ReadTimeout("sensitive URL or credentials", request=request)
        result, _, _ = await self.probe(handler=timeout)
        self.assertFalse(result[0])
        self.assertNotIn("sensitive", result[1])

    async def test_embedding_model_needs_successful_embedding_operation(self):
        payload = health()
        payload["models"] = {"qwen3-embedding:0.6b": copy.deepcopy(payload["models"]["qwen3:8b"])}
        self.assertFalse((await self.probe(payload, model="qwen3-embedding:0.6b"))[0][0])
        payload["models"]["qwen3-embedding:0.6b"]["last_success_operation"] = "embed"
        self.assertTrue((await self.probe(payload, model="qwen3-embedding:0.6b"))[0][0])


if __name__ == "__main__":
    unittest.main()
