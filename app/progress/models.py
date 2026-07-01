"""Relational model for the evidence-linked research progress ledger."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.progress.database import Base


def uuid_string() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class Member(Base, TimestampMixin):
    __tablename__ = "progress_members"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    clerk_user_id: Mapped[str | None] = mapped_column(String(255), unique=True)
    name_original: Mapped[str] = mapped_column(String(500))
    name_en: Mapped[str | None] = mapped_column(String(500))
    name_ja: Mapped[str | None] = mapped_column(String(500))
    role: Mapped[str | None] = mapped_column(String(255))
    affiliation: Mapped[str | None] = mapped_column(String(500))
    notes: Mapped[str | None] = mapped_column(Text)


class Meeting(Base, TimestampMixin):
    __tablename__ = "progress_meetings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    title_original: Mapped[str] = mapped_column(String(1000))
    title_en: Mapped[str | None] = mapped_column(String(1000))
    title_ja: Mapped[str | None] = mapped_column(String(1000))
    meeting_date: Mapped[date] = mapped_column(Date, index=True)
    location: Mapped[str | None] = mapped_column(String(500))
    meeting_type: Mapped[str] = mapped_column(String(64), default="other")
    summary_en: Mapped[str | None] = mapped_column(Text)
    summary_ja: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)

    sessions: Mapped[list[MeetingSession]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan", order_by="MeetingSession.order_index"
    )
    presentations: Mapped[list[Presentation]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan"
    )


class MeetingSession(Base, TimestampMixin):
    __tablename__ = "progress_meeting_sessions"
    __table_args__ = (UniqueConstraint("meeting_id", "order_index"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    meeting_id: Mapped[str] = mapped_column(
        ForeignKey("progress_meetings.id", ondelete="CASCADE"), index=True
    )
    title_original: Mapped[str] = mapped_column(String(1000))
    title_en: Mapped[str | None] = mapped_column(String(1000))
    title_ja: Mapped[str | None] = mapped_column(String(1000))
    session_type: Mapped[str] = mapped_column(String(64), default="unknown")
    start_time: Mapped[str | None] = mapped_column(String(32))
    end_time: Mapped[str | None] = mapped_column(String(32))
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str | None] = mapped_column(Text)

    meeting: Mapped[Meeting] = relationship(back_populates="sessions")


class Presentation(Base, TimestampMixin):
    __tablename__ = "progress_presentations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    meeting_id: Mapped[str] = mapped_column(
        ForeignKey("progress_meetings.id", ondelete="CASCADE"), index=True
    )
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("progress_meeting_sessions.id", ondelete="SET NULL")
    )
    presenter_member_id: Mapped[str | None] = mapped_column(
        ForeignKey("progress_members.id", ondelete="SET NULL")
    )
    title_original: Mapped[str] = mapped_column(String(1000))
    title_en: Mapped[str | None] = mapped_column(String(1000))
    title_ja: Mapped[str | None] = mapped_column(String(1000))
    language_primary: Mapped[str] = mapped_column(String(16), default="unknown")
    start_time: Mapped[str | None] = mapped_column(String(32))
    end_time: Mapped[str | None] = mapped_column(String(32))
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    summary_en: Mapped[str | None] = mapped_column(Text)
    summary_ja: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)

    meeting: Mapped[Meeting] = relationship(back_populates="presentations")


class RepositoryMaterialLink(Base, TimestampMixin):
    __tablename__ = "progress_repository_material_links"
    __table_args__ = (
        UniqueConstraint("meeting_id", "repository_material_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    meeting_id: Mapped[str] = mapped_column(
        ForeignKey("progress_meetings.id", ondelete="CASCADE"), index=True
    )
    presentation_id: Mapped[str | None] = mapped_column(
        ForeignKey("progress_presentations.id", ondelete="SET NULL")
    )
    repository_material_id: Mapped[str] = mapped_column(String(36), index=True)
    link_status: Mapped[str] = mapped_column(String(32), default="linked")
    notes: Mapped[str | None] = mapped_column(Text)


class TranscriptSegment(Base, TimestampMixin):
    __tablename__ = "progress_transcript_segments"
    __table_args__ = (
        UniqueConstraint("repository_material_id", "source_locator"),
        Index("idx_progress_transcript_meeting_start", "meeting_id", "start_seconds"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    meeting_id: Mapped[str] = mapped_column(
        ForeignKey("progress_meetings.id", ondelete="CASCADE"), index=True
    )
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("progress_meeting_sessions.id", ondelete="SET NULL")
    )
    presentation_id: Mapped[str | None] = mapped_column(
        ForeignKey("progress_presentations.id", ondelete="SET NULL")
    )
    speaker_text: Mapped[str | None] = mapped_column(String(500))
    speaker_member_id: Mapped[str | None] = mapped_column(
        ForeignKey("progress_members.id", ondelete="SET NULL")
    )
    start_seconds: Mapped[float | None] = mapped_column(Float)
    end_seconds: Mapped[float | None] = mapped_column(Float)
    segment_type: Mapped[str] = mapped_column(String(64), default="unknown")
    original_text: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(16), default="unknown")
    translation_en: Mapped[str | None] = mapped_column(Text)
    translation_ja: Mapped[str | None] = mapped_column(Text)
    summary_en: Mapped[str | None] = mapped_column(Text)
    summary_ja: Mapped[str | None] = mapped_column(Text)
    repository_material_id: Mapped[str | None] = mapped_column(String(36), index=True)
    repository_segment_id: Mapped[int | None] = mapped_column(Integer)
    source_locator: Mapped[str] = mapped_column(String(500))
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text)


class ResearchRecord(Base, TimestampMixin):
    __tablename__ = "progress_research_records"
    __table_args__ = (
        Index("idx_progress_records_type_status", "record_type", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    record_type: Mapped[str] = mapped_column(String(64), index=True)
    title_original: Mapped[str | None] = mapped_column(String(1000))
    title_en: Mapped[str | None] = mapped_column(String(1000))
    title_ja: Mapped[str | None] = mapped_column(String(1000))
    summary_en: Mapped[str | None] = mapped_column(Text)
    summary_ja: Mapped[str | None] = mapped_column(Text)
    original_evidence: Mapped[str | None] = mapped_column(Text)
    original_language: Mapped[str] = mapped_column(String(16), default="unknown")
    meeting_id: Mapped[str | None] = mapped_column(
        ForeignKey("progress_meetings.id", ondelete="SET NULL"), index=True
    )
    presentation_id: Mapped[str | None] = mapped_column(
        ForeignKey("progress_presentations.id", ondelete="SET NULL")
    )
    member_id: Mapped[str | None] = mapped_column(
        ForeignKey("progress_members.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_by: Mapped[str] = mapped_column(String(255))
    reviewed_by: Mapped[str | None] = mapped_column(String(255))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)

    evidence_links: Mapped[list[RecordEvidence]] = relationship(
        back_populates="record", cascade="all, delete-orphan"
    )


class EvidenceCitation(Base, TimestampMixin):
    __tablename__ = "progress_evidence_citations"
    __table_args__ = (
        Index("idx_progress_evidence_repository", "repository_material_id", "repository_segment_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    source_kind: Mapped[str] = mapped_column(String(64), index=True)
    transcript_segment_id: Mapped[str | None] = mapped_column(
        ForeignKey("progress_transcript_segments.id", ondelete="SET NULL")
    )
    repository_material_id: Mapped[str | None] = mapped_column(String(36))
    repository_segment_id: Mapped[int | None] = mapped_column(Integer)
    repository_image_id: Mapped[str | None] = mapped_column(String(36))
    external_url: Mapped[str | None] = mapped_column(String(2000))
    source_locator: Mapped[str | None] = mapped_column(String(500))
    citation_text_original: Mapped[str | None] = mapped_column(Text)
    citation_text_en: Mapped[str | None] = mapped_column(Text)
    citation_text_ja: Mapped[str | None] = mapped_column(Text)
    source_hash: Mapped[str | None] = mapped_column(String(128))
    notes: Mapped[str | None] = mapped_column(Text)


class RecordEvidence(Base):
    __tablename__ = "progress_record_evidence"

    record_id: Mapped[str] = mapped_column(
        ForeignKey("progress_research_records.id", ondelete="CASCADE"), primary_key=True
    )
    evidence_id: Mapped[str] = mapped_column(
        ForeignKey("progress_evidence_citations.id", ondelete="CASCADE"), primary_key=True
    )
    relation: Mapped[str] = mapped_column(String(32), default="supports")
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    record: Mapped[ResearchRecord] = relationship(back_populates="evidence_links")
    evidence: Mapped[EvidenceCitation] = relationship()


class Concept(Base, TimestampMixin):
    __tablename__ = "progress_concepts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    label_en: Mapped[str] = mapped_column(String(500), index=True)
    label_ja: Mapped[str] = mapped_column(String(500), index=True)
    aliases_en: Mapped[list] = mapped_column(JSON, default=list)
    aliases_ja: Mapped[list] = mapped_column(JSON, default=list)
    definition_en: Mapped[str | None] = mapped_column(Text)
    definition_ja: Mapped[str | None] = mapped_column(Text)
    concept_type: Mapped[str] = mapped_column(String(64), default="other")
    source_vocabulary: Mapped[str | None] = mapped_column(String(500))
    source_version: Mapped[str | None] = mapped_column(String(128))


class RecordConcept(Base):
    __tablename__ = "progress_record_concepts"

    record_id: Mapped[str] = mapped_column(
        ForeignKey("progress_research_records.id", ondelete="CASCADE"), primary_key=True
    )
    concept_id: Mapped[str] = mapped_column(
        ForeignKey("progress_concepts.id", ondelete="CASCADE"), primary_key=True
    )


class LocalizedRevision(Base):
    __tablename__ = "progress_localized_revisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    entity_type: Mapped[str] = mapped_column(String(64), index=True)
    entity_id: Mapped[str] = mapped_column(String(36), index=True)
    field_name: Mapped[str] = mapped_column(String(64))
    language: Mapped[str] = mapped_column(String(16))
    previous_text: Mapped[str | None] = mapped_column(Text)
    revised_text: Mapped[str | None] = mapped_column(Text)
    review_status: Mapped[str] = mapped_column(String(32), default="candidate")
    changed_by: Mapped[str] = mapped_column(String(255))
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AuditEvent(Base):
    __tablename__ = "progress_audit_events"
    __table_args__ = (Index("idx_progress_audit_entity", "entity_type", "entity_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    entity_type: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[str] = mapped_column(String(36))
    action: Mapped[str] = mapped_column(String(64))
    actor_id: Mapped[str] = mapped_column(String(255))
    changes: Mapped[dict] = mapped_column(JSON, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ProgressJob(Base, TimestampMixin):
    __tablename__ = "progress_jobs"
    __table_args__ = (Index("idx_progress_jobs_status_type", "status", "job_type"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    job_type: Mapped[str] = mapped_column(String(64))
    target_id: Mapped[str] = mapped_column(String(255), index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    queued_by: Mapped[str] = mapped_column(String(255))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SourceAlignment(Base, TimestampMixin):
    __tablename__ = "progress_source_alignments"
    __table_args__ = (
        UniqueConstraint(
            "transcript_segment_id",
            "repository_material_id",
            "repository_segment_id",
            name="uq_progress_source_alignment",
        ),
        Index("idx_progress_alignment_meeting", "meeting_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    meeting_id: Mapped[str] = mapped_column(
        ForeignKey("progress_meetings.id", ondelete="CASCADE"), index=True
    )
    presentation_id: Mapped[str | None] = mapped_column(
        ForeignKey("progress_presentations.id", ondelete="SET NULL")
    )
    transcript_segment_id: Mapped[str] = mapped_column(
        ForeignKey("progress_transcript_segments.id", ondelete="CASCADE"), index=True
    )
    repository_material_id: Mapped[str] = mapped_column(String(36), index=True)
    repository_segment_id: Mapped[int | None] = mapped_column(Integer)
    method: Mapped[str] = mapped_column(String(32), default="manual")
    confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="candidate")
    created_by: Mapped[str] = mapped_column(String(255))
    reviewed_by: Mapped[str | None] = mapped_column(String(255))
    notes: Mapped[str | None] = mapped_column(Text)


class AiExtractionRun(Base, TimestampMixin):
    __tablename__ = "progress_ai_extraction_runs"
    __table_args__ = (Index("idx_progress_ai_run_status", "status", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(255))
    prompt_version: Mapped[str] = mapped_column(String(128))
    prompt_hash: Mapped[str] = mapped_column(String(128))
    source_kind: Mapped[str] = mapped_column(String(64))
    source_ids: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="running", index=True)
    raw_response: Mapped[str | None] = mapped_column(Text)
    candidate_record_ids: Mapped[list] = mapped_column(JSON, default=list)
    requested_by: Mapped[str] = mapped_column(String(255))
    error_message: Mapped[str | None] = mapped_column(Text)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class NarrativeArc(Base, TimestampMixin):
    __tablename__ = "progress_narrative_arcs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    title_en: Mapped[str] = mapped_column(String(1000))
    title_ja: Mapped[str] = mapped_column(String(1000))
    description_en: Mapped[str] = mapped_column(Text)
    description_ja: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    notes: Mapped[str | None] = mapped_column(Text)

    links: Mapped[list[ArcLink]] = relationship(
        back_populates="arc", cascade="all, delete-orphan", order_by="ArcLink.order_index"
    )


class ArcLink(Base, TimestampMixin):
    __tablename__ = "progress_arc_links"
    __table_args__ = (UniqueConstraint("arc_id", "source_kind", "source_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    arc_id: Mapped[str] = mapped_column(
        ForeignKey("progress_narrative_arcs.id", ondelete="CASCADE"), index=True
    )
    source_kind: Mapped[str] = mapped_column(String(64))
    source_id: Mapped[str] = mapped_column(String(36))
    role_in_arc: Mapped[str] = mapped_column(String(32), default="evidence")
    summary_en: Mapped[str | None] = mapped_column(Text)
    summary_ja: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str | None] = mapped_column(Text)

    arc: Mapped[NarrativeArc] = relationship(back_populates="links")
