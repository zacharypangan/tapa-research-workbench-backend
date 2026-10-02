import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api import repository
from app.auth import ROLE_PERMISSIONS, _context_from_claims
from app.progress import worker
from app.progress.models import EvidenceCitation, ProgressJob
from app.repository import links
from app.repository.schemas import ExtractRequest, ExtractionResult
import test.test_progress as progress_tests


class ProductionFixTests(unittest.TestCase):
    def setUp(self):
        self.fixture = progress_tests.ProgressApiTests()
        self.fixture.setUp()
        self.client = self.fixture.client
        self.client.app.include_router(repository.router, prefix="/api/v1")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = patch.multiple(repository, STORAGE_ROOT=str(self.root), DB_PATH=str(self.root / "repository.sqlite"), FILES_ROOT=str(self.root / "files"), IMAGES_ROOT=str(self.root / "images"), SessionLocal=self.fixture.sessions)
        self.paths.start()
        repository.init_repository_db()
        self.worker_sessions = patch.object(worker, "SessionLocal", self.fixture.sessions)
        self.worker_sessions.start()

    def tearDown(self):
        self.worker_sessions.stop()
        self.paths.stop()
        self.fixture.tearDown()
        self.temporary.cleanup()

    def material(self, upload=False):
        response = self.client.post("/api/v1/repository/materials", json={"title": "Synthetic source"})
        self.assertEqual(response.status_code, 200, response.text)
        material_id = response.json()["id"]
        if upload:
            response = self.client.post(f"/api/v1/repository/materials/{material_id}/files", params={"filename": "source.txt", "process": "false"}, content=b"Synthetic evidence for a production regression test.")
            self.assertEqual(response.status_code, 200, response.text)
        return material_id

    def extract(self, material_id, **options):
        return asyncio.run(repository.extract_material_text(material_id, ExtractRequest(include_links=False), **options))

    def segment(self, material_id):
        with repository.get_connection() as con:
            return dict(con.execute("SELECT * FROM extracted_segments WHERE material_id = ?", (material_id,)).fetchone())

    def test_read_only_user_cannot_create_delete_or_review(self):
        material_id = self.material()
        from app.auth import AuthContext, get_auth_context
        self.client.app.dependency_overrides[get_auth_context] = lambda: AuthContext("viewer", "org-1", "viewer", frozenset(ROLE_PERMISSIONS["viewer"]), {})
        self.assertEqual(self.client.get("/api/v1/repository/materials").status_code, 200)
        self.assertEqual(self.client.post("/api/v1/repository/materials", json={"title": "Denied"}).status_code, 403)
        self.assertEqual(self.client.delete(f"/api/v1/repository/materials/{material_id}").status_code, 403)
        self.assertEqual(self.client.patch("/api/v1/repository/graph/edges/fake/review", json={"review_status": "accepted"}).status_code, 403)
        self.assertEqual(self.client.post("/api/v1/repository/ai/semantic-search", json={"query": "source", "auto_index": True}).status_code, 403)
        self.assertEqual(self.client.post("/api/v1/repository/ai/multimodal-search", json={"query": "source", "auto_index_images": True}).status_code, 403)
        self.assertEqual(self.client.post("/api/v1/repository/ai/evidence-report", json={"query": "source", "auto_index": True}).status_code, 403)

    def test_indexing_skips_blank_images_without_deleting_evidence(self):
        material_id = self.material()
        image_path = self.root / "synthetic-image.png"
        image_path.write_bytes(b"synthetic source")
        with repository.get_connection() as con:
            con.execute("INSERT INTO image_evidence(id, material_id, evidence_type, source_kind, source_locator, page_ref, page_index, image_path, extraction_method, created_at) VALUES ('image-1', ?, 'image', 'file', 'source.pdf', '1', 1, ?, 'synthetic', '2026-10-02')", (material_id, str(image_path)))
            with patch.object(repository, "ai_embedding_configured", return_value=True), patch.object(repository, "is_informative_image", return_value=False):
                result = asyncio.run(repository.index_missing_image_embeddings(con, material_id))
            self.assertEqual(result["skipped_blank_count"], 1)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM image_evidence").fetchone()[0], 1)
        self.assertTrue(image_path.exists())

    def test_member_cannot_create_reviewed_records(self):
        self.fixture.role = "member"
        for state in ("accepted", "rejected", "superseded"):
            response = self.client.post("/api/v1/progress/records", json={"record_type": "decision", "status": state, "evidence": [{"source_kind": "note", "citation_text_original": "Test note"}]})
            self.assertEqual(response.status_code, 403, response.text)

    def test_reviewer_created_acceptance_records_reviewer(self):
        self.fixture.role = "reviewer"
        response = self.client.post("/api/v1/progress/records", json={"record_type": "decision", "status": "accepted", "evidence": [{"source_kind": "note", "citation_text_original": "Test note"}]})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["reviewed_by"], "reviewer-1")
        self.assertIsNotNone(response.json()["reviewed_at"])

    def test_failed_and_empty_replacement_preserve_evidence(self):
        material_id = self.material(upload=True)
        self.extract(material_id)
        before = self.segment(material_id)
        with patch.object(repository, "extract_text_from_file", side_effect=RuntimeError("Parser failed")):
            with self.assertRaises(RuntimeError):
                self.extract(material_id, force=True)
        self.assertEqual(self.segment(material_id), before)
        with patch.object(repository, "extract_text_from_file", return_value=ExtractionResult(segments=[], warnings=[])):
            with self.assertRaises(HTTPException) as failure:
                self.extract(material_id, force=True)
            self.assertEqual(failure.exception.status_code, 422)
        self.assertEqual(self.segment(material_id), before)

    def test_failed_replacement_transaction_rolls_back_deletions(self):
        material_id = self.material(upload=True)
        self.extract(material_id)
        before = self.segment(material_id)
        with repository.get_connection() as con:
            con.execute("CREATE TRIGGER fail_replacement BEFORE INSERT ON extracted_segments BEGIN SELECT RAISE(ABORT, 'Synthetic insert failure'); END")
        with self.assertRaises(Exception):
            self.extract(material_id, force=True)
        self.assertEqual(self.segment(material_id), before)

    def test_observation_and_progress_citations_prevent_replacement(self):
        material_id = self.material(upload=True)
        self.extract(material_id)
        before = self.segment(material_id)
        response = self.client.post(f"/api/v1/repository/materials/{material_id}/observations", json={"observed_text": "Synthetic", "source_segment_id": before["id"]})
        self.assertEqual(response.status_code, 200, response.text)
        with self.assertRaises(HTTPException) as failure:
            self.extract(material_id, force=True)
        self.assertEqual(failure.exception.status_code, 409)
        self.assertEqual(self.segment(material_id), before)
        with repository.get_connection() as con:
            con.execute("DELETE FROM observations WHERE material_id = ?", (material_id,))
        with self.fixture.sessions() as session:
            session.add(EvidenceCitation(source_kind="repository_segment", repository_material_id=material_id, repository_segment_id=before["id"]))
            session.commit()
        self.assertEqual(self.client.delete(f"/api/v1/repository/materials/{material_id}").status_code, 409)
        with self.assertRaises(HTTPException) as failure:
            self.extract(material_id, force=True)
        self.assertEqual(failure.exception.status_code, 409)

    def test_replayed_extraction_keeps_segment_identifiers(self):
        material_id = self.material(upload=True)
        first = self.extract(material_id, run_id="synthetic-job")
        before = self.segment(material_id)
        with patch.object(repository, "extract_text_from_file", side_effect=AssertionError("Replay must not parse again")):
            second = self.extract(material_id, run_id="synthetic-job")
        self.assertEqual(second, first)
        self.assertEqual(self.segment(material_id), before)

    def test_upload_stream_limit_and_deduplication(self):
        material_id = self.material()
        path = f"/api/v1/repository/materials/{material_id}/files?filename=source.txt&process=false"
        with patch.object(repository, "MAX_UPLOAD_BYTES", 5):
            response = self.client.post(path, content=iter([b"123", b"456"]))
        self.assertEqual(response.status_code, 413)
        self.assertEqual(list((self.root / "files" / material_id).iterdir()), [])
        first = self.client.post(path, content=b"same source").json()
        second = self.client.post(path, content=b"same source").json()
        self.assertEqual(first["id"], second["id"])
        self.assertTrue(second["deduplicated"])
        self.assertEqual(len(list((self.root / "files" / material_id).iterdir())), 1)

    def test_pagination_exposes_all_records_and_global_status_counts(self):
        with repository.get_connection() as con:
            con.executemany("INSERT INTO materials(id, title, source_type, status, created_at, updated_at) VALUES (?, ?, 'other', 'needs_metadata', '2026-10-02', '2026-10-02')", [(f"synthetic-{i}", f"Source {i}") for i in range(105)])
        first = self.client.get("/api/v1/repository/materials?limit=100").json()
        second = self.client.get("/api/v1/repository/materials?limit=100&offset=100").json()
        self.assertEqual(first["total"], 105)
        self.assertEqual(first["status_counts"]["needs_metadata"], 105)
        self.assertEqual(len(first["materials"]), 100)
        self.assertEqual(len(second["materials"]), 5)
        self.assertFalse({item["id"] for item in first["materials"]} & {item["id"] for item in second["materials"]})

    def test_extraction_route_queues_instead_of_parsing(self):
        material_id = self.material(upload=True)
        pool = AsyncMock()
        with patch("app.repository.jobs.create_pool", new=AsyncMock(return_value=pool)), patch.object(repository, "extract_text_from_file", side_effect=AssertionError("API must not parse")):
            response = self.client.post(f"/api/v1/repository/materials/{material_id}/extract", json={"include_links": False})
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["job_id"]
        pool.enqueue_job.assert_awaited_once_with("extract_repository_material", job_id, _job_id=job_id)
        self.assertEqual(self.client.get(f"/api/v1/progress/jobs/{job_id}").json()["status"], "queued")

    def test_queue_failure_has_persisted_failure_status(self):
        material_id = self.material()
        with patch("app.repository.jobs.create_pool", new=AsyncMock(side_effect=RuntimeError("Queue down"))):
            response = self.client.post(f"/api/v1/repository/materials/{material_id}/extract", json={})
        self.assertEqual(response.status_code, 503)
        with self.fixture.sessions() as session:
            job = session.get(ProgressJob, response.json()["detail"]["job_id"])
            self.assertEqual(job.status, "queue_failed")

    def test_worker_runs_image_work_and_replays_completed_result(self):
        with self.fixture.sessions() as session:
            job = ProgressJob(job_type="repository_extract", target_id="synthetic", queued_by="admin")
            session.add(job)
            session.commit()
            job_id = job.id
        extract = AsyncMock(return_value={"image_evidence_count": 1, "warnings": []})
        index = AsyncMock(return_value={"indexed_count": 1})
        text_index = AsyncMock(return_value={"indexed_count": 1})
        with patch.object(worker, "extract_material_text", extract), patch.object(worker, "build_image_index", index), patch.object(worker, "build_semantic_index", text_index):
            first = asyncio.run(worker.extract_repository_material({}, job_id))
            second = asyncio.run(worker.extract_repository_material({}, job_id))
        extract.assert_awaited_once()
        index.assert_awaited_once()
        text_index.assert_awaited_once()
        self.assertEqual(first, second)
        self.assertEqual(self.client.get(f"/api/v1/progress/jobs/{job_id}").json()["status"], "completed")

    def test_worker_restart_recovers_persisted_jobs(self):
        with self.fixture.sessions() as session:
            session.add(ProgressJob(job_type="repository_extract", target_id="synthetic", queued_by="admin", status="queued"))
            session.commit()
            job_id = session.scalar(select(ProgressJob.id))
        redis = AsyncMock()
        asyncio.run(worker.startup({"redis": redis}))
        redis.enqueue_job.assert_awaited_once_with("extract_repository_material", job_id, _job_id=job_id)

    def test_legacy_routes_are_not_mounted_and_vercel_cors_is_restricted(self):
        import main
        client = TestClient(main.app)
        for path in ("/api/v1/data", "/api/v1/catalog", "/api/v1/models", "/api/v1/tiles/type/data/0/0/0.pbf"):
            self.assertEqual(client.get(path).status_code, 404)
        self.assertEqual(client.post("/api/v1/chat", json={}).status_code, 404)
        self.assertEqual(client.options("/api/v1/repository/materials", headers={"Origin": "https://unrelated-review.vercel.app", "Access-Control-Request-Method": "GET"}).status_code, 400)

    def test_readiness_requires_storage_and_worker_heartbeat(self):
        import main
        redis = AsyncMock()
        client = TestClient(main.app)
        with patch.object(main, "SessionLocal", self.fixture.sessions), patch.object(main.Redis, "from_url", return_value=redis):
            redis.get.return_value = None
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 503)
            redis.get.return_value = b"worker heartbeat"
            ready = client.get("/ready")
            self.assertEqual(ready.status_code, 200, ready.text)
            self.assertTrue(all(ready.json()["checks"].values()))

    def test_production_configuration_cannot_enable_development_auth(self):
        import main
        with patch.dict(os.environ, {"APP_ENV": "production", "AUTH_DISABLED": "true"}):
            with self.assertRaises(HTTPException):
                main.on_startup()
        with patch.dict(os.environ, {"AUTH_DISABLED": "false", "CLERK_ORGANIZATION_ID": ""}):
            with self.assertRaises(RuntimeError):
                main.on_startup()


class ProjectMembershipTests(unittest.TestCase):
    def test_missing_wrong_and_unknown_organization_roles_are_denied(self):
        with patch.dict(os.environ, {"CLERK_ORGANIZATION_ID": "allowed-org"}):
            for claims in ({"sub": "user"}, {"sub": "user", "o": {"id": "other-org", "rol": "admin"}}, {"sub": "user", "o": {"id": "allowed-org", "rol": "guest"}}):
                with self.assertRaises(HTTPException) as failure:
                    _context_from_claims(claims)
                self.assertEqual(failure.exception.status_code, 403)
            context = _context_from_claims({"sub": "user", "o": {"id": "allowed-org", "rol": "viewer"}, "role": "admin", "org_permissions": ["progress:admin"]})
            self.assertEqual(context.permissions, frozenset({"progress:read"}))


class SafeLinkTests(unittest.IsolatedAsyncioTestCase):
    async def test_unapproved_and_private_destinations_are_rejected(self):
        with patch.dict(os.environ, {"REPOSITORY_ALLOWED_LINK_HOSTS": "example.org"}):
            with self.assertRaises(ValueError):
                await links.checked_url("http://127.0.0.1/")
            addresses = [(socket_family, 1, 6, "", ("127.0.0.1", 443)) for socket_family in [2]]
            with patch.object(links.socket, "getaddrinfo", return_value=addresses):
                with self.assertRaises(ValueError):
                    await links.checked_url("https://example.org/")

    async def test_redirect_is_checked_before_second_request(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(302, headers={"Location": "http://127.0.0.1/"})
        with patch.dict(os.environ, {"REPOSITORY_ALLOWED_LINK_HOSTS": "example.org"}), patch.object(links.socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaises(ValueError):
                    await links.fetch_html(client, "https://example.org/")
        self.assertEqual(len(requests), 1)

    async def test_page_size_limit_is_enforced(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, headers={"Content-Type": "text/html"}, content=b"0123456789"))
        with patch.dict(os.environ, {"REPOSITORY_ALLOWED_LINK_HOSTS": "example.org"}), patch.object(links.socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]), patch.object(links, "MAX_PAGE_BYTES", 5):
            async with httpx.AsyncClient(transport=transport) as client:
                with self.assertRaises(ValueError):
                    await links.fetch_html(client, "https://example.org/")


class ServiceSupervisorTests(unittest.IsolatedAsyncioTestCase):
    async def check_shutdown(self, signal_shutdown):
        from app import service
        loop = asyncio.get_running_loop()
        callbacks = []

        class Process:
            def __init__(self, exited=False):
                self.returncode = 1 if exited else None
                self.done = asyncio.Event()
                if exited:
                    self.done.set()
                self.terminated = False

            async def wait(self):
                await self.done.wait()
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15
                self.done.set()

        processes = [Process(exited=not signal_shutdown), Process()]
        calls = 0

        async def start(*arguments):
            nonlocal calls
            process = processes[calls]
            calls += 1
            if signal_shutdown and calls == 2:
                loop.call_soon(callbacks[0])
            return process

        with patch.object(loop, "add_signal_handler", side_effect=lambda signum, callback: callbacks.append(callback)), patch.object(service.asyncio, "create_subprocess_exec", side_effect=start):
            result = await service.serve()
        self.assertEqual(result, 0 if signal_shutdown else 1)
        self.assertTrue(processes[1].terminated)
        self.assertTrue(all(process.returncode is not None for process in processes))

    async def test_child_exit_stops_other_process_and_fails_service(self):
        await self.check_shutdown(False)

    async def test_shutdown_signal_stops_both_processes(self):
        await self.check_shutdown(True)


if __name__ == "__main__":
    unittest.main()
