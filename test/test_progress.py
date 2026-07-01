import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import progress
from app.auth import AuthContext, get_auth_context
from app.progress.database import Base, get_session
from app.progress.models import AuditEvent


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


if __name__ == "__main__":
    unittest.main()
