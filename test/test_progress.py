import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import progress
from app.auth import AuthContext, get_auth_context
from app.progress.database import Base, get_session
from app.progress.models import (
    AiExtractionRun,
    AuditEvent,
    LocalizedRevision,
    ResearchRecord,
    TranscriptSegment,
)


class ProgressApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False, class_=Session)
        self.role = "admin"

        def session_override():
            with self.sessions() as session:
                yield session

        def auth_override():
            permissions = {
                "member": {"progress:read", "progress:write"},
                "reviewer": {"progress:read", "progress:write", "progress:review"},
                "admin": {
                    "progress:read",
                    "progress:write",
                    "progress:review",
                    "progress:admin",
                },
            }[self.role]
            return AuthContext(
                user_id=f"{self.role}-1",
                organization_id="org-1",
                role=self.role,
                permissions=frozenset(permissions),
                claims={"sub": f"{self.role}-1"},
            )

        app = FastAPI()
        app.include_router(progress.router, prefix="/api/v1")
        app.dependency_overrides[get_session] = session_override
        app.dependency_overrides[get_auth_context] = auth_override
        self.client = TestClient(app)

    def tearDown(self):
        self.engine.dispose()

    def create_meeting(self):
        response = self.client.post(
            "/api/v1/progress/meetings",
            json={
                "title_original": "国際先導研究会",
                "title_en": "International Research Meeting",
                "meeting_date": "2025-04-27",
                "meeting_type": "hybrid",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_member_and_meeting_crud_create_audit_events(self):
        member = self.client.post(
            "/api/v1/progress/members",
            json={"name_original": "菊澤律子", "name_en": "Ritsuko Kikusawa"},
        )
        self.assertEqual(member.status_code, 201, member.text)
        meeting = self.create_meeting()

        detail = self.client.get(f"/api/v1/progress/meetings/{meeting['id']}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["meeting"]["title_original"], "国際先導研究会")

        with self.sessions() as session:
            actions = list(session.scalars(select(AuditEvent.action).order_by(AuditEvent.occurred_at)))
        self.assertEqual(actions, ["created", "created"])

    def test_accepted_record_requires_evidence(self):
        meeting = self.create_meeting()
        response = self.client.post(
            "/api/v1/progress/records",
            json={
                "record_type": "decision",
                "title_en": "Preserve original text",
                "meeting_id": meeting["id"],
                "status": "accepted",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Accepted records require evidence.")

    def test_candidate_can_be_reviewed_with_evidence(self):
        meeting = self.create_meeting()
        response = self.client.post(
            "/api/v1/progress/records",
            json={
                "record_type": "temporal_issue",
                "title_en": "Collection year requires explicit metadata",
                "title_ja": "採録年には明示的なメタデータが必要",
                "meeting_id": meeting["id"],
                "status": "candidate",
                "evidence": [
                    {
                        "source_kind": "note",
                        "citation_text_original": "時間情報には属性が必要である。",
                        "source_locator": "review-note:1",
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        record = response.json()
        self.assertEqual(len(record["evidence"]), 1)

        self.role = "reviewer"
        review = self.client.post(
            f"/api/v1/progress/records/{record['id']}/review",
            json={"decision": "accepted"},
        )
        self.assertEqual(review.status_code, 200, review.text)
        self.assertEqual(review.json()["status"], "accepted")

    def test_member_cannot_open_review_queue(self):
        self.role = "member"
        response = self.client.get("/api/v1/progress/review-queue")
        self.assertEqual(response.status_code, 403)

    def test_extraction_job_is_persisted_before_queueing(self):
        pool = AsyncMock()
        pool.enqueue_job = AsyncMock()
        pool.close = AsyncMock()
        with patch("app.api.progress.create_pool", new=AsyncMock(return_value=pool)):
            response = self.client.post(
                "/api/v1/progress/jobs/repository-extract/material-1?force=true"
            )
        self.assertEqual(response.status_code, 202, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "queued")
        self.assertEqual(payload["payload"], {"force": True})
        pool.enqueue_job.assert_awaited_once_with(
            "extract_repository_material", payload["id"], _job_id=payload["id"]
        )

    def test_bilingual_concept_alias_search_returns_original_evidence(self):
        meeting = self.create_meeting()
        concept = self.client.post(
            "/api/v1/progress/concepts",
            json={
                "label_en": "collection year",
                "label_ja": "採録年",
                "aliases_en": ["temporal metadata"],
                "aliases_ja": ["時間情報"],
                "concept_type": "temporal",
            },
        )
        self.assertEqual(concept.status_code, 201, concept.text)
        record = self.client.post(
            "/api/v1/progress/records",
            json={
                "record_type": "temporal_issue",
                "title_original": "採録年の区別",
                "original_evidence": "採録年と発行年を区別する必要がある。",
                "meeting_id": meeting["id"],
                "status": "candidate",
                "concept_ids": [concept.json()["id"]],
            },
        )
        self.assertEqual(record.status_code, 201, record.text)

        response = self.client.get("/api/v1/progress/search", params={"q": "collection year"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()[0]["original_text"], "採録年と発行年を区別する必要がある。")
        self.assertEqual(response.json()[0]["retrieval_basis"], "concept_alias")

    def test_transcript_translation_creates_revision_history(self):
        meeting = self.create_meeting()
        with self.sessions() as session:
            segment = TranscriptSegment(
                meeting_id=meeting["id"],
                original_text="時間情報が必要です。",
                language="ja",
                source_locator="cue:1",
            )
            session.add(segment)
            session.commit()
            segment_id = segment.id

        response = self.client.patch(
            f"/api/v1/progress/transcript-segments/{segment_id}",
            json={"translation_en": "Temporal information is required.", "reviewed": True},
        )
        self.assertEqual(response.status_code, 200, response.text)
        with self.sessions() as session:
            revisions = list(session.scalars(select(LocalizedRevision)))
        self.assertEqual(len(revisions), 1)
        self.assertEqual(revisions[0].review_status, "reviewed")

    def test_ai_extraction_can_only_create_evidence_linked_candidates(self):
        meeting = self.create_meeting()
        with self.sessions() as session:
            segment = TranscriptSegment(
                meeting_id=meeting["id"],
                original_text="GIS should be used as an analytical interface.",
                language="en",
                source_locator="cue:2",
            )
            session.add(segment)
            session.commit()
            segment_id = segment.id
        generated = [
            {
                "record_type": "method_claim",
                "title_original": "GIS as analytical interface",
                "title_en": "GIS as analytical interface",
                "title_ja": None,
                "summary_en": "GIS supports analysis rather than display alone.",
                "summary_ja": None,
                "confidence": 0.87,
                "evidence_source_ids": [segment_id],
                "details": {"requested_status": "accepted"},
            }
        ]
        with patch(
            "app.api.progress.request_candidate_extraction",
            new=AsyncMock(return_value=("[]", generated)),
        ):
            response = self.client.post(
                "/api/v1/progress/ai/extract",
                json={"source_kind": "transcript_segment", "source_ids": [segment_id]},
            )
        self.assertEqual(response.status_code, 201, response.text)
        with self.sessions() as session:
            run = session.scalars(select(AiExtractionRun)).one()
            record = session.scalars(select(ResearchRecord)).one()
        self.assertEqual(run.status, "completed")
        self.assertEqual(record.status, "candidate")
        self.assertEqual(len(response.json()["candidate_record_ids"]), 1)

    def test_meeting_and_narrative_reports_preserve_evidence(self):
        meeting = self.create_meeting()
        record_response = self.client.post(
            "/api/v1/progress/records",
            json={
                "record_type": "decision",
                "title_en": "Keep originals primary",
                "meeting_id": meeting["id"],
                "status": "candidate",
                "evidence": [
                    {
                        "source_kind": "note",
                        "citation_text_original": "原文を一次資料として保存する。",
                        "source_locator": "meeting-note:3",
                    }
                ],
            },
        )
        self.assertEqual(record_response.status_code, 201, record_response.text)
        record = record_response.json()
        arc_response = self.client.post(
            "/api/v1/progress/narrative-arcs",
            json={
                "title_en": "From data collection to datafication",
                "title_ja": "データ収集からデータ化へ",
                "description_en": "How materials become structured evidence.",
                "description_ja": "資料が構造化された証拠になる過程。",
            },
        )
        self.assertEqual(arc_response.status_code, 201, arc_response.text)
        arc = arc_response.json()
        link_response = self.client.post(
            f"/api/v1/progress/narrative-arcs/{arc['id']}/links",
            json={"source_kind": "research_record", "source_id": record["id"], "role_in_arc": "decision"},
        )
        self.assertEqual(link_response.status_code, 201, link_response.text)

        meeting_report = self.client.get(f"/api/v1/progress/reports/meeting/{meeting['id']}")
        arc_report = self.client.get(f"/api/v1/progress/reports/narrative-arcs/{arc['id']}")
        self.assertIn("meeting-note:3", meeting_report.json()["content"])
        self.assertIn("Keep originals primary", arc_report.json()["content"])


if __name__ == "__main__":
    unittest.main()
