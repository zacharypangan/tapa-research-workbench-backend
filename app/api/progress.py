"""Evidence-linked Research Progress Portal API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload
from arq import create_pool
from arq.connections import RedisSettings
import os

from app.auth import AuthContext, require_permission
from app.progress.database import get_session
from app.progress.models import (
    AuditEvent,
    Concept,
    EvidenceCitation,
    Meeting,
    MeetingSession,
    Member,
    Presentation,
    ProgressJob,
    RecordConcept,
    RecordEvidence,
    RepositoryMaterialLink,
    ResearchRecord,
    TranscriptSegment,
)
from app.progress.schemas import (
    ConceptCreate,
    ConceptOut,
    EvidenceOut,
    MaterialLinkCreate,
    MeetingCreate,
    MeetingOut,
    MeetingUpdate,
    MemberCreate,
    MemberOut,
    MemberUpdate,
    PresentationCreate,
    ResearchRecordCreate,
    ResearchRecordOut,
    ResearchRecordUpdate,
    ReviewRequest,
    SessionCreate,
)


router = APIRouter(prefix="/progress", tags=["progress"])


def _not_found(entity: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{entity} not found.")


def _commit(session: Session, conflict_message: str = "Record already exists.") -> None:
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=conflict_message) from exc


def _audit(
    session: Session,
    *,
    entity_type: str,
    entity_id: str,
    action: str,
    actor_id: str,
    changes: dict[str, Any] | None = None,
) -> None:
    session.add(
        AuditEvent(
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            actor_id=actor_id,
            changes=changes or {},
        )
    )


def _apply_updates(instance: Any, updates: dict[str, Any]) -> dict[str, Any]:
    changed: dict[str, Any] = {}
    for field, value in updates.items():
        previous = getattr(instance, field)
        if previous != value:
            changed[field] = {"before": previous, "after": value}
            setattr(instance, field, value)
    return changed


def _record_out(record: ResearchRecord) -> ResearchRecordOut:
    evidence = [EvidenceOut.model_validate(link.evidence) for link in record.evidence_links]
    return ResearchRecordOut.model_validate(record).model_copy(update={"evidence": evidence})


@router.get("/status")
def progress_status(
    context: AuthContext = Depends(require_permission("progress:read")),
):
    return {
        "available": True,
        "user_id": context.user_id,
        "role": context.role,
        "permissions": sorted(context.permissions),
    }


@router.get("/members", response_model=list[MemberOut])
def list_members(
    query: str = Query(default="", max_length=500),
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    statement = select(Member).order_by(Member.name_original)
    if query.strip():
        term = f"%{query.strip()}%"
        statement = statement.where(
            or_(Member.name_original.ilike(term), Member.name_en.ilike(term), Member.name_ja.ilike(term))
        )
    return list(session.scalars(statement))


@router.post("/members", response_model=MemberOut, status_code=status.HTTP_201_CREATED)
def create_member(
    payload: MemberCreate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    member = Member(**payload.model_dump())
    session.add(member)
    session.flush()
    _audit(session, entity_type="member", entity_id=member.id, action="created", actor_id=context.user_id)
    _commit(session, "A member already uses that Clerk user ID.")
    session.refresh(member)
    return member


@router.patch("/members/{member_id}", response_model=MemberOut)
def update_member(
    member_id: str,
    payload: MemberUpdate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    member = session.get(Member, member_id)
    if not member:
        raise _not_found("Member")
    changes = _apply_updates(member, payload.model_dump(exclude_unset=True))
    _audit(
        session,
        entity_type="member",
        entity_id=member.id,
        action="updated",
        actor_id=context.user_id,
        changes=changes,
    )
    _commit(session)
    session.refresh(member)
    return member


@router.get("/meetings", response_model=list[MeetingOut])
def list_meetings(
    query: str = Query(default="", max_length=500),
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    statement = select(Meeting).order_by(Meeting.meeting_date.desc())
    if query.strip():
        term = f"%{query.strip()}%"
        statement = statement.where(
            or_(Meeting.title_original.ilike(term), Meeting.title_en.ilike(term), Meeting.title_ja.ilike(term))
        )
    return list(session.scalars(statement))


@router.post("/meetings", response_model=MeetingOut, status_code=status.HTTP_201_CREATED)
def create_meeting(
    payload: MeetingCreate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    meeting = Meeting(**payload.model_dump())
    session.add(meeting)
    session.flush()
    _audit(session, entity_type="meeting", entity_id=meeting.id, action="created", actor_id=context.user_id)
    _commit(session)
    session.refresh(meeting)
    return meeting


@router.patch("/meetings/{meeting_id}", response_model=MeetingOut)
def update_meeting(
    meeting_id: str,
    payload: MeetingUpdate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    meeting = session.get(Meeting, meeting_id)
    if not meeting:
        raise _not_found("Meeting")
    changes = _apply_updates(meeting, payload.model_dump(exclude_unset=True))
    _audit(
        session,
        entity_type="meeting",
        entity_id=meeting.id,
        action="updated",
        actor_id=context.user_id,
        changes=changes,
    )
    _commit(session)
    session.refresh(meeting)
    return meeting


@router.get("/meetings/{meeting_id}")
def meeting_detail(
    meeting_id: str,
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    meeting = session.get(Meeting, meeting_id)
    if not meeting:
        raise _not_found("Meeting")
    records = list(
        session.scalars(
            select(ResearchRecord)
            .where(ResearchRecord.meeting_id == meeting_id)
            .options(selectinload(ResearchRecord.evidence_links).selectinload(RecordEvidence.evidence))
            .order_by(ResearchRecord.created_at.desc())
        )
    )
    sessions = list(
        session.scalars(
            select(MeetingSession)
            .where(MeetingSession.meeting_id == meeting_id)
            .order_by(MeetingSession.order_index)
        )
    )
    presentations = list(
        session.scalars(
            select(Presentation)
            .where(Presentation.meeting_id == meeting_id)
            .order_by(Presentation.order_index)
        )
    )
    materials = list(
        session.scalars(
            select(RepositoryMaterialLink).where(RepositoryMaterialLink.meeting_id == meeting_id)
        )
    )
    transcript_segments = list(
        session.scalars(
            select(TranscriptSegment)
            .where(TranscriptSegment.meeting_id == meeting_id)
            .order_by(TranscriptSegment.start_seconds)
        )
    )
    return {
        "meeting": MeetingOut.model_validate(meeting),
        "sessions": sessions,
        "presentations": presentations,
        "materials": materials,
        "transcript_segments": transcript_segments,
        "records": [_record_out(record) for record in records],
    }


@router.post("/meetings/{meeting_id}/sessions", status_code=status.HTTP_201_CREATED)
def create_session(
    meeting_id: str,
    payload: SessionCreate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    if not session.get(Meeting, meeting_id):
        raise _not_found("Meeting")
    meeting_session = MeetingSession(meeting_id=meeting_id, **payload.model_dump())
    session.add(meeting_session)
    session.flush()
    _audit(
        session,
        entity_type="meeting_session",
        entity_id=meeting_session.id,
        action="created",
        actor_id=context.user_id,
    )
    _commit(session, "That session order is already used for this meeting.")
    session.refresh(meeting_session)
    return meeting_session


@router.post("/meetings/{meeting_id}/presentations", status_code=status.HTTP_201_CREATED)
def create_presentation(
    meeting_id: str,
    payload: PresentationCreate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    if not session.get(Meeting, meeting_id):
        raise _not_found("Meeting")
    presentation = Presentation(meeting_id=meeting_id, **payload.model_dump())
    session.add(presentation)
    session.flush()
    _audit(
        session,
        entity_type="presentation",
        entity_id=presentation.id,
        action="created",
        actor_id=context.user_id,
    )
    _commit(session)
    session.refresh(presentation)
    return presentation


@router.post("/meetings/{meeting_id}/materials", status_code=status.HTTP_201_CREATED)
def link_material(
    meeting_id: str,
    payload: MaterialLinkCreate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    if not session.get(Meeting, meeting_id):
        raise _not_found("Meeting")
    link = RepositoryMaterialLink(meeting_id=meeting_id, **payload.model_dump())
    session.add(link)
    session.flush()
    _audit(
        session,
        entity_type="repository_material_link",
        entity_id=link.id,
        action="created",
        actor_id=context.user_id,
        changes={"repository_material_id": link.repository_material_id},
    )
    _commit(session, "That Repository material is already linked to this meeting.")
    session.refresh(link)
    return link


@router.get("/records", response_model=list[ResearchRecordOut])
def list_records(
    meeting_id: str | None = None,
    record_type: str | None = None,
    record_status: str | None = Query(default=None, alias="status"),
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    statement = (
        select(ResearchRecord)
        .options(selectinload(ResearchRecord.evidence_links).selectinload(RecordEvidence.evidence))
        .order_by(ResearchRecord.updated_at.desc())
    )
    if meeting_id:
        statement = statement.where(ResearchRecord.meeting_id == meeting_id)
    if record_type:
        statement = statement.where(ResearchRecord.record_type == record_type)
    if record_status:
        statement = statement.where(ResearchRecord.status == record_status)
    return [_record_out(record) for record in session.scalars(statement)]


@router.get("/review-queue", response_model=list[ResearchRecordOut])
def review_queue(
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:review")),
):
    statement = (
        select(ResearchRecord)
        .where(ResearchRecord.status.in_(["candidate", "needs_review"]))
        .options(selectinload(ResearchRecord.evidence_links).selectinload(RecordEvidence.evidence))
        .order_by(ResearchRecord.created_at)
    )
    return [_record_out(record) for record in session.scalars(statement)]


@router.post("/records", response_model=ResearchRecordOut, status_code=status.HTTP_201_CREATED)
def create_record(
    payload: ResearchRecordCreate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    if payload.status == "accepted" and not payload.evidence:
        raise HTTPException(status_code=422, detail="Accepted records require evidence.")
    record_data = payload.model_dump(exclude={"evidence", "concept_ids"})
    record = ResearchRecord(**record_data, created_by=context.user_id)
    session.add(record)
    session.flush()
    for index, evidence_payload in enumerate(payload.evidence):
        evidence = EvidenceCitation(**evidence_payload.model_dump())
        session.add(evidence)
        session.flush()
        session.add(
            RecordEvidence(
                record_id=record.id,
                evidence_id=evidence.id,
                relation="supports",
                order_index=index,
            )
        )
    for concept_id in dict.fromkeys(payload.concept_ids):
        if not session.get(Concept, concept_id):
            raise HTTPException(status_code=422, detail=f"Concept not found: {concept_id}")
        session.add(RecordConcept(record_id=record.id, concept_id=concept_id))
    _audit(
        session,
        entity_type="research_record",
        entity_id=record.id,
        action="created",
        actor_id=context.user_id,
        changes={"status": record.status, "record_type": record.record_type},
    )
    _commit(session)
    statement = (
        select(ResearchRecord)
        .where(ResearchRecord.id == record.id)
        .options(selectinload(ResearchRecord.evidence_links).selectinload(RecordEvidence.evidence))
    )
    return _record_out(session.scalars(statement).one())


@router.patch("/records/{record_id}", response_model=ResearchRecordOut)
def update_record(
    record_id: str,
    payload: ResearchRecordUpdate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    statement = (
        select(ResearchRecord)
        .where(ResearchRecord.id == record_id)
        .options(selectinload(ResearchRecord.evidence_links).selectinload(RecordEvidence.evidence))
    )
    record = session.scalars(statement).one_or_none()
    if not record:
        raise _not_found("Research record")
    if record.status == "accepted" and "progress:review" not in context.permissions:
        raise HTTPException(status_code=403, detail="Only reviewers may edit accepted records.")
    changes = _apply_updates(record, payload.model_dump(exclude_unset=True))
    _audit(
        session,
        entity_type="research_record",
        entity_id=record.id,
        action="updated",
        actor_id=context.user_id,
        changes=changes,
    )
    _commit(session)
    session.refresh(record)
    return _record_out(record)


@router.post("/records/{record_id}/review", response_model=ResearchRecordOut)
def review_record(
    record_id: str,
    payload: ReviewRequest,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:review")),
):
    statement = (
        select(ResearchRecord)
        .where(ResearchRecord.id == record_id)
        .options(selectinload(ResearchRecord.evidence_links).selectinload(RecordEvidence.evidence))
    )
    record = session.scalars(statement).one_or_none()
    if not record:
        raise _not_found("Research record")
    if payload.decision == "accepted" and not record.evidence_links:
        raise HTTPException(status_code=422, detail="Accepted records require evidence.")
    previous = record.status
    record.status = payload.decision
    record.reviewed_by = context.user_id
    record.reviewed_at = datetime.now(timezone.utc)
    if payload.notes:
        record.notes = payload.notes
    _audit(
        session,
        entity_type="research_record",
        entity_id=record.id,
        action="reviewed",
        actor_id=context.user_id,
        changes={"status": {"before": previous, "after": payload.decision}},
    )
    _commit(session)
    session.refresh(record)
    return _record_out(record)


@router.get("/concepts", response_model=list[ConceptOut])
def list_concepts(
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    return list(session.scalars(select(Concept).order_by(Concept.label_en)))


@router.post("/concepts", response_model=ConceptOut, status_code=status.HTTP_201_CREATED)
def create_concept(
    payload: ConceptCreate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:admin")),
):
    concept = Concept(**payload.model_dump())
    session.add(concept)
    session.flush()
    _audit(
        session,
        entity_type="concept",
        entity_id=concept.id,
        action="created",
        actor_id=context.user_id,
    )
    _commit(session)
    session.refresh(concept)
    return concept


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    job = session.get(ProgressJob, job_id)
    if not job:
        raise _not_found("Progress job")
    return job


@router.post("/jobs/repository-extract/{material_id}", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_repository_extraction(
    material_id: str,
    force: bool = False,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    job = ProgressJob(
        job_type="repository_extract",
        target_id=material_id,
        status="queued",
        payload={"force": force},
        queued_by=context.user_id,
    )
    session.add(job)
    session.flush()
    _audit(
        session,
        entity_type="progress_job",
        entity_id=job.id,
        action="queued",
        actor_id=context.user_id,
        changes={"material_id": material_id, "force": force},
    )
    _commit(session)

    redis_url = os.getenv("REDIS_PRIVATE_URL") or os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        pool = await create_pool(RedisSettings.from_dsn(redis_url))
        await pool.enqueue_job("extract_repository_material", job.id, _job_id=job.id)
        await pool.close()
    except Exception as exc:
        job.status = "queue_failed"
        job.error_message = str(exc)
        _commit(session)
        raise HTTPException(
            status_code=503,
            detail={"message": "Extraction queue is unavailable.", "job_id": job.id},
        ) from exc
    session.refresh(job)
    return job
