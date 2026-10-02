import os
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from datetime import date
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import progress
from app.auth import AuthContext, get_auth_context
from app.progress.archive_structure import LinkedMaterialSnapshot, backfill_meeting_structure
from app.progress.database import Base, get_session
from app.progress.models import (
    AiExtractionRun,
    AuditEvent,
    LocalizedRevision,
    Meeting,
    MeetingSession,
    Presentation,
    RepositoryMaterialLink,
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

    def create_repository_fixture(self, rows):
        handle = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        handle.close()
        con = sqlite3.connect(handle.name)
        try:
            con.execute(
                """
                CREATE TABLE materials (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    source_type TEXT,
                    status TEXT,
                    raw_reference TEXT
                )
                """
            )
            con.execute(
                """
                CREATE TABLE files (
                    id TEXT PRIMARY KEY,
                    material_id TEXT,
                    original_filename TEXT,
                    parser_status TEXT,
                    parser_message TEXT
                )
                """
            )
            for row in rows:
                con.execute(
                    "INSERT INTO materials (id, title, source_type, status, raw_reference) VALUES (?, ?, ?, ?, ?)",
                    (
                        row["id"],
                        row.get("title"),
                        row.get("source_type"),
                        row.get("status", "ready_for_text_extraction"),
                        row.get("raw_reference"),
                    ),
                )
                con.execute(
                    "INSERT INTO files (id, material_id, original_filename, parser_status, parser_message) VALUES (?, ?, ?, ?, ?)",
                    (
                        f"file-{row['id']}",
                        row["id"],
                        row.get("original_filename"),
                        row.get("parser_status", "pending"),
                        row.get("parser_message"),
                    ),
                )
            con.commit()
        finally:
            con.close()
        return handle.name

    @contextmanager
    def repository_connection_override(self, path):
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()

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

    def test_archive_structure_backfill_is_idempotent_and_groups_variants(self):
        with self.sessions() as session:
            meeting = Meeting(
                title_original="国際先導研究会",
                title_en="International Research Meeting",
                meeting_date=date(2025, 4, 27),
                meeting_type="hybrid",
            )
            session.add(meeting)
            session.flush()
            links = [
                RepositoryMaterialLink(meeting_id=meeting.id, repository_material_id="mat-pdf", link_status="linked"),
                RepositoryMaterialLink(meeting_id=meeting.id, repository_material_id="mat-pptx", link_status="linked"),
                RepositoryMaterialLink(meeting_id=meeting.id, repository_material_id="mat-transcript", link_status="linked"),
            ]
            session.add_all(links)
            session.commit()
            snapshots = [
                LinkedMaterialSnapshot(
                    link_id=links[0].id,
                    repository_material_id="mat-pdf",
                    repository_title="20250427_Data Collection in Fiji v1.0",
                    original_filename="20250427_Data Collection in Fiji v1.0.pdf",
                    source_type="pdf",
                    raw_reference="Presentation Materials/20250427/20250427_Data Collection in Fiji v1.0.pdf",
                    parser_status="pending",
                ),
                LinkedMaterialSnapshot(
                    link_id=links[1].id,
                    repository_material_id="mat-pptx",
                    repository_title="20250427_Data Collection in Fiji v1.1",
                    original_filename="20250427_Data Collection in Fiji v1.1.pptx",
                    source_type="presentation",
                    raw_reference="Presentation Materials/20250427/20250427_Data Collection in Fiji v1.1.pptx",
                    parser_status="pending",
                ),
                LinkedMaterialSnapshot(
                    link_id=links[2].id,
                    repository_material_id="mat-transcript",
                    repository_title="20250427_Recording",
                    original_filename="20250427_Recording.transcript.vtt",
                    source_type="workshop_material",
                    raw_reference="Minutes/20250427_Recording.transcript.vtt",
                    parser_status="pending",
                ),
            ]

        with self.sessions() as session:
            managed_meeting = session.get(Meeting, meeting.id)
            first_summary = backfill_meeting_structure(session, managed_meeting, snapshots)
            session.commit()
        with self.sessions() as session:
            managed_meeting = session.get(Meeting, meeting.id)
            second_summary = backfill_meeting_structure(session, managed_meeting, snapshots)
            session.commit()
            sessions = list(session.scalars(select(MeetingSession)))
            presentations = list(session.scalars(select(Presentation)))
            links = list(session.scalars(select(RepositoryMaterialLink).order_by(RepositoryMaterialLink.created_at)))

        self.assertEqual(first_summary["sessions_created"], 1)
        self.assertEqual(first_summary["presentations_created"], 1)
        self.assertEqual(first_summary["material_assignments_created"], 2)
        self.assertEqual(second_summary["sessions_created"], 0)
        self.assertEqual(second_summary["presentations_created"], 0)
        self.assertEqual(second_summary["material_assignments_created"], 0)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(len(presentations), 1)
        self.assertEqual(sum(1 for link in links if link.presentation_id), 2)
        self.assertIsNone(links[-1].presentation_id)

    def test_patch_endpoints_update_structure_and_audit(self):
        meeting = self.create_meeting()
        session_response = self.client.post(
            f"/api/v1/progress/meetings/{meeting['id']}/sessions",
            json={"title_original": "Morning session", "order_index": 0},
        )
        self.assertEqual(session_response.status_code, 201, session_response.text)
        other_session_response = self.client.post(
            f"/api/v1/progress/meetings/{meeting['id']}/sessions",
            json={"title_original": "Afternoon session", "order_index": 1},
        )
        self.assertEqual(other_session_response.status_code, 201, other_session_response.text)
        presentation_response = self.client.post(
            f"/api/v1/progress/meetings/{meeting['id']}/presentations",
            json={"title_original": "Collection year metadata", "session_id": session_response.json()["id"]},
        )
        self.assertEqual(presentation_response.status_code, 201, presentation_response.text)
        link_response = self.client.post(
            f"/api/v1/progress/meetings/{meeting['id']}/materials",
            json={"repository_material_id": "material-1"},
        )
        self.assertEqual(link_response.status_code, 201, link_response.text)

        updated_session = self.client.patch(
            f"/api/v1/progress/sessions/{session_response.json()['id']}",
            json={"title_original": "Morning plenary", "order_index": 0},
        )
        updated_presentation = self.client.patch(
            f"/api/v1/progress/presentations/{presentation_response.json()['id']}",
            json={"session_id": other_session_response.json()["id"], "title_original": "Collection year review"},
        )
        updated_link = self.client.patch(
            f"/api/v1/progress/material-links/{link_response.json()['id']}",
            json={"presentation_id": presentation_response.json()["id"], "notes": "Attached during archive structuring."},
        )
        self.assertEqual(updated_session.status_code, 200, updated_session.text)
        self.assertEqual(updated_presentation.status_code, 200, updated_presentation.text)
        self.assertEqual(updated_link.status_code, 200, updated_link.text)

        with self.sessions() as session:
            actions = list(
                session.scalars(
                    select(AuditEvent.action).where(AuditEvent.action == "updated")
                )
            )
        self.assertGreaterEqual(len(actions), 3)

    def test_member_cannot_open_review_queue(self):
        self.role = "member"
        response = self.client.get("/api/v1/progress/review-queue")
        self.assertEqual(response.status_code, 403)

    def test_extraction_job_is_persisted_before_queueing(self):
        pool = AsyncMock()
        pool.enqueue_job = AsyncMock()
        pool.close = AsyncMock()
        with patch("app.repository.jobs.create_pool", new=AsyncMock(return_value=pool)):
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
        meeting_detail = self.client.get(f"/api/v1/progress/meetings/{meeting['id']}")
        self.assertIn("meeting-note:3", meeting_report.json()["content"])
        self.assertIn("Keep originals primary", arc_report.json()["content"])
        self.assertEqual(meeting_detail.status_code, 200, meeting_detail.text)
        self.assertEqual(meeting_detail.json()["structure"]["related_arc_count"], 1)
        self.assertEqual(meeting_detail.json()["related_arcs"][0]["current_meeting_steps"], 1)

    def test_dashboard_tracks_project_progress_and_arc_development(self):
        meeting = self.create_meeting()
        session_response = self.client.post(
            f"/api/v1/progress/meetings/{meeting['id']}/sessions",
            json={"title_original": "Foundations", "order_index": 0},
        )
        self.assertEqual(session_response.status_code, 201, session_response.text)
        presentation_response = self.client.post(
            f"/api/v1/progress/meetings/{meeting['id']}/presentations",
            json={
                "title_original": "Execution plan review",
                "session_id": session_response.json()["id"],
                "order_index": 0,
            },
        )
        self.assertEqual(presentation_response.status_code, 201, presentation_response.text)
        material_response = self.client.post(
            f"/api/v1/progress/meetings/{meeting['id']}/materials",
            json={"repository_material_id": "material-1"},
        )
        self.assertEqual(material_response.status_code, 201, material_response.text)
        with self.sessions() as session:
            session.add(
                TranscriptSegment(
                    meeting_id=meeting["id"],
                    original_text="Narrative arcs should remain evidence-linked.",
                    language="en",
                    source_locator="cue:1",
                )
            )
            session.commit()

        record_response = self.client.post(
            "/api/v1/progress/records",
            json={
                "record_type": "decision",
                "title_en": "Keep arcs anchored in evidence",
                "meeting_id": meeting["id"],
                "status": "accepted",
                "evidence": [
                    {
                        "source_kind": "note",
                        "citation_text_original": "Narrative arcs should remain evidence-linked.",
                        "source_locator": "execution-plan:narrative",
                    }
                ],
            },
        )
        self.assertEqual(record_response.status_code, 201, record_response.text)
        record = record_response.json()

        arc_response = self.client.post(
            "/api/v1/progress/narrative-arcs",
            json={
                "title_en": "Governed narrative development",
                "title_ja": "統治されたナラティブ展開",
                "description_en": "Progress records accumulate into inspectable project arcs.",
                "description_ja": "進捗レコードが検証可能なプロジェクト・アークへ積み上がる。",
            },
        )
        self.assertEqual(arc_response.status_code, 201, arc_response.text)
        arc = arc_response.json()
        arc_link_response = self.client.post(
            f"/api/v1/progress/narrative-arcs/{arc['id']}/links",
            json={
                "source_kind": "research_record",
                "source_id": record["id"],
                "role_in_arc": "origin",
                "summary_en": "The execution plan requires narrative to remain evidence-linked.",
            },
        )
        self.assertEqual(arc_link_response.status_code, 201, arc_link_response.text)

        response = self.client.get("/api/v1/progress/dashboard")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["summary"]["meeting_count"], 1)
        self.assertEqual(payload["summary"]["session_count"], 1)
        self.assertEqual(payload["summary"]["presentation_count"], 1)
        self.assertEqual(payload["summary"]["material_count"], 1)
        self.assertEqual(payload["summary"]["accepted_record_count"], 1)
        self.assertEqual(payload["summary"]["narrative_arc_count"], 1)
        self.assertEqual(payload["summary"]["transcript_segment_count"], 1)
        self.assertGreaterEqual(len(payload["narrative_treatment"]), 4)
        self.assertEqual(len(payload["timeline"]), 1)
        self.assertEqual(payload["timeline"][0]["meeting"]["meeting_date"], "2025-04-27")
        self.assertEqual(payload["timeline"][0]["record_count"], 1)
        self.assertEqual(payload["timeline"][0]["narrative_arc_step_count"], 1)
        self.assertIn("Keep arcs anchored in evidence", payload["timeline"][0]["featured_record_titles"])
        self.assertIn("Governed narrative development", payload["timeline"][0]["arc_titles"])
        self.assertEqual(payload["arcs"][0]["links"][0]["meeting_date"], "2025-04-27")

    def test_transcript_only_meeting_detail_reports_structure(self):
        meeting = self.create_meeting()
        with self.sessions() as session:
            session.add(MeetingSession(meeting_id=meeting["id"], title_original="Meeting materials", order_index=0))
            session.add(
                RepositoryMaterialLink(
                    meeting_id=meeting["id"],
                    repository_material_id="transcript-material",
                    link_status="linked",
                )
            )
            session.commit()
        repository_path = self.create_repository_fixture(
            [
                {
                    "id": "transcript-material",
                    "title": "20250427_Recording",
                    "source_type": "workshop_material",
                    "raw_reference": "Minutes/20250427_Recording.transcript.vtt",
                    "original_filename": "20250427_Recording.transcript.vtt",
                }
            ]
        )
        try:
            with patch(
                "app.api.progress.get_repository_connection",
                new=lambda: self.repository_connection_override(repository_path),
            ):
                response = self.client.get(f"/api/v1/progress/meetings/{meeting['id']}")
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertTrue(payload["structure"]["transcript_only"])
            self.assertEqual(payload["structure"]["presentation_count"], 0)
            self.assertTrue(payload["materials"][0]["is_transcript_source"])
        finally:
            os.unlink(repository_path)


if __name__ == "__main__":
    unittest.main()
