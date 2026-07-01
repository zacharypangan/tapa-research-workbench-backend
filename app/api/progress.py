"""Evidence-linked Research Progress Portal API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload
from arq import create_pool
from arq.connections import RedisSettings
import os

from app.auth import AuthContext, require_permission
from app.api.repository import get_connection as get_repository_connection
from app.progress.database import get_session
from app.progress.models import (
    AuditEvent,
    AiExtractionRun,
    Concept,
    EvidenceCitation,
    Meeting,
    MeetingSession,
    Member,
    Presentation,
    ProgressJob,
    LocalizedRevision,
    RecordConcept,
    RecordEvidence,
    RepositoryMaterialLink,
    ResearchRecord,
    NarrativeArc,
    ArcLink,
    SourceAlignment,
    TranscriptSegment,
)
from app.progress.ai import (
    PROMPT_VERSION,
    build_candidate_prompt,
    prompt_hash,
    provider_metadata,
    request_candidate_extraction,
)
from app.progress.reports import (
    meeting_report,
    narrative_arc_report,
    record_report,
    repository_enrichment_csv,
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
    AlignmentCreate,
    AlignmentOut,
    SearchResult,
    SessionCreate,
    TranscriptSegmentUpdate,
    AiExtractionRequest,
    ArcLinkCreate,
    NarrativeArcCreate,
    NarrativeArcOut,
    ReportPayload,
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
    repository_metadata: dict[str, dict[str, Any]] = {}
    material_ids = [link.repository_material_id for link in materials]
    if material_ids:
        placeholders = ", ".join("?" for _ in material_ids)
        with get_repository_connection() as repository_connection:
            rows = repository_connection.execute(
                f"""
                SELECT m.id, m.title, m.source_type, m.status,
                       f.original_filename, f.parser_status, f.parser_message
                FROM materials m
                LEFT JOIN files f ON f.material_id = m.id
                WHERE m.id IN ({placeholders})
                """,
                material_ids,
            ).fetchall()
        repository_metadata = {row["id"]: dict(row) for row in rows}
    material_payload = []
    for link in materials:
        metadata = repository_metadata.get(link.repository_material_id, {})
        material_payload.append(
            {
                "id": link.id,
                "repository_material_id": link.repository_material_id,
                "link_status": link.link_status,
                "notes": link.notes,
                "repository_title": metadata.get("title"),
                "original_filename": metadata.get("original_filename"),
                "source_type": metadata.get("source_type"),
                "repository_status": metadata.get("status"),
                "parser_status": metadata.get("parser_status"),
                "parser_message": metadata.get("parser_message"),
            }
        )
    transcript_segments = list(
        session.scalars(
            select(TranscriptSegment)
            .where(TranscriptSegment.meeting_id == meeting_id)
            .order_by(TranscriptSegment.start_seconds)
            .limit(100)
        )
    )
    transcript_count = session.scalar(
        select(func.count()).select_from(TranscriptSegment).where(TranscriptSegment.meeting_id == meeting_id)
    ) or 0
    return {
        "meeting": MeetingOut.model_validate(meeting),
        "sessions": sessions,
        "presentations": presentations,
        "materials": material_payload,
        "transcript_segments": transcript_segments,
        "transcript_count": transcript_count,
        "records": [_record_out(record) for record in records],
    }


@router.get("/meetings/{meeting_id}/transcript")
def meeting_transcript(
    meeting_id: str,
    q: str = Query(default="", max_length=500),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    if not session.get(Meeting, meeting_id):
        raise _not_found("Meeting")
    filters = [TranscriptSegment.meeting_id == meeting_id]
    if q.strip():
        term = f"%{q.strip()}%"
        filters.append(
            or_(
                TranscriptSegment.original_text.ilike(term),
                TranscriptSegment.translation_en.ilike(term),
                TranscriptSegment.translation_ja.ilike(term),
                TranscriptSegment.speaker_text.ilike(term),
            )
        )
    total = session.scalar(
        select(func.count()).select_from(TranscriptSegment).where(*filters)
    ) or 0
    segments = list(
        session.scalars(
            select(TranscriptSegment)
            .where(*filters)
            .order_by(TranscriptSegment.start_seconds, TranscriptSegment.created_at)
            .limit(limit)
            .offset(offset)
        )
    )
    return {"segments": segments, "total": total, "limit": limit, "offset": offset}


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


@router.patch("/transcript-segments/{segment_id}")
def update_transcript_segment(
    segment_id: str,
    payload: TranscriptSegmentUpdate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    segment = session.get(TranscriptSegment, segment_id)
    if not segment:
        raise _not_found("Transcript segment")
    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        previous = getattr(segment, field)
        if previous == value:
            continue
        if field in {"translation_en", "translation_ja", "summary_en", "summary_ja"}:
            session.add(
                LocalizedRevision(
                    entity_type="transcript_segment",
                    entity_id=segment.id,
                    field_name=field,
                    language="ja" if field.endswith("_ja") else "en",
                    previous_text=previous,
                    revised_text=value,
                    review_status="reviewed" if payload.reviewed else "candidate",
                    changed_by=context.user_id,
                )
            )
        setattr(segment, field, value)
    _audit(
        session,
        entity_type="transcript_segment",
        entity_id=segment.id,
        action="updated",
        actor_id=context.user_id,
        changes={"fields": sorted(updates)},
    )
    _commit(session)
    session.refresh(segment)
    return segment


@router.get("/alignments", response_model=list[AlignmentOut])
def list_alignments(
    meeting_id: str | None = None,
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    statement = select(SourceAlignment).order_by(SourceAlignment.created_at.desc())
    if meeting_id:
        statement = statement.where(SourceAlignment.meeting_id == meeting_id)
    return list(session.scalars(statement))


@router.post("/alignments", response_model=AlignmentOut, status_code=status.HTTP_201_CREATED)
def create_alignment(
    payload: AlignmentCreate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    if not session.get(Meeting, payload.meeting_id):
        raise _not_found("Meeting")
    segment = session.get(TranscriptSegment, payload.transcript_segment_id)
    if not segment or segment.meeting_id != payload.meeting_id:
        raise HTTPException(status_code=422, detail="Transcript segment does not belong to the meeting.")
    alignment = SourceAlignment(**payload.model_dump(), created_by=context.user_id)
    if payload.status == "accepted" and "progress:review" in context.permissions:
        alignment.reviewed_by = context.user_id
    session.add(alignment)
    session.flush()
    _audit(
        session,
        entity_type="source_alignment",
        entity_id=alignment.id,
        action="created",
        actor_id=context.user_id,
        changes={"method": alignment.method, "confidence": alignment.confidence},
    )
    _commit(session, "That source alignment already exists.")
    session.refresh(alignment)
    return alignment


def _normalized(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def _match_result(
    *,
    terms: list[tuple[str, str]],
    original: str,
    translated: str,
) -> tuple[str, str, float] | None:
    normalized_original = _normalized(original)
    normalized_translation = _normalized(translated)
    for term, basis in terms:
        normalized_term = _normalized(term)
        if normalized_term and normalized_term in normalized_original:
            return term, "original" if basis == "query" else "concept_alias", 1.0 if basis == "query" else 0.82
        if normalized_term and normalized_term in normalized_translation:
            return term, "translation" if basis == "query" else "concept_alias", 0.9 if basis == "query" else 0.78
    return None


@router.get("/search", response_model=list[SearchResult])
def bilingual_search(
    q: str = Query(min_length=1, max_length=500),
    meeting_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    query = _normalized(q)
    terms: list[tuple[str, str]] = [(q, "query")]
    for concept in session.scalars(select(Concept)):
        labels = [concept.label_en, concept.label_ja, *concept.aliases_en, *concept.aliases_ja]
        if any(query in _normalized(label) or _normalized(label) in query for label in labels if label):
            terms.extend((label, "concept") for label in labels if label)
    terms = list(dict.fromkeys(terms))

    record_statement = select(ResearchRecord).order_by(ResearchRecord.updated_at.desc())
    segment_statement = select(TranscriptSegment).order_by(TranscriptSegment.start_seconds)
    if meeting_id:
        record_statement = record_statement.where(ResearchRecord.meeting_id == meeting_id)
        segment_statement = segment_statement.where(TranscriptSegment.meeting_id == meeting_id)

    results: list[SearchResult] = []
    for record in session.scalars(record_statement):
        original = " ".join(filter(None, [record.title_original, record.original_evidence]))
        translated = " ".join(
            filter(None, [record.title_en, record.title_ja, record.summary_en, record.summary_ja])
        )
        match = _match_result(terms=terms, original=original, translated=translated)
        if match:
            results.append(
                SearchResult(
                    source_kind="research_record",
                    source_id=record.id,
                    meeting_id=record.meeting_id,
                    title=record.title_original or record.title_en or record.title_ja,
                    original_text=record.original_evidence,
                    translation_en=record.title_en,
                    translation_ja=record.title_ja,
                    summary_en=record.summary_en,
                    summary_ja=record.summary_ja,
                    source_locator=None,
                    matched_term=match[0],
                    retrieval_basis=match[1],
                    score=match[2],
                )
            )
    for segment in session.scalars(segment_statement):
        translated = " ".join(
            filter(None, [segment.translation_en, segment.translation_ja, segment.summary_en, segment.summary_ja])
        )
        match = _match_result(terms=terms, original=segment.original_text, translated=translated)
        if match:
            results.append(
                SearchResult(
                    source_kind="transcript_segment",
                    source_id=segment.id,
                    meeting_id=segment.meeting_id,
                    title=segment.speaker_text,
                    original_text=segment.original_text,
                    translation_en=segment.translation_en,
                    translation_ja=segment.translation_ja,
                    summary_en=segment.summary_en,
                    summary_ja=segment.summary_ja,
                    source_locator=segment.source_locator,
                    matched_term=match[0],
                    retrieval_basis=match[1],
                    score=match[2],
                )
            )
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:limit]


@router.post("/ai/extract", status_code=status.HTTP_201_CREATED)
async def extract_ai_candidates(
    payload: AiExtractionRequest,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    segments = list(
        session.scalars(
            select(TranscriptSegment).where(TranscriptSegment.id.in_(payload.source_ids))
        )
    )
    if len(segments) != len(set(payload.source_ids)):
        raise HTTPException(status_code=422, detail="One or more transcript segments were not found.")
    sources = [
        {"id": segment.id, "language": segment.language, "text": segment.original_text}
        for segment in segments
    ]
    prompt = build_candidate_prompt(sources)
    provider, model = provider_metadata()
    run = AiExtractionRun(
        provider=provider,
        model=model,
        prompt_version=PROMPT_VERSION,
        prompt_hash=prompt_hash(prompt),
        source_kind=payload.source_kind,
        source_ids=payload.source_ids,
        status="running",
        requested_by=context.user_id,
    )
    session.add(run)
    session.flush()
    try:
        raw_response, candidates = await request_candidate_extraction(sources)
        segment_by_id = {segment.id: segment for segment in segments}
        record_ids: list[str] = []
        for candidate in candidates:
            record = ResearchRecord(
                record_type=candidate["record_type"],
                title_original=candidate.get("title_original"),
                title_en=candidate.get("title_en"),
                title_ja=candidate.get("title_ja"),
                summary_en=candidate.get("summary_en"),
                summary_ja=candidate.get("summary_ja"),
                original_language="mixed",
                meeting_id=segments[0].meeting_id,
                status="candidate",
                confidence=candidate["confidence"],
                details={
                    **candidate["details"],
                    "ai_run_id": run.id,
                    "prompt_version": PROMPT_VERSION,
                },
                created_by=context.user_id,
            )
            session.add(record)
            session.flush()
            for order_index, source_id in enumerate(dict.fromkeys(candidate["evidence_source_ids"])):
                source = segment_by_id.get(source_id)
                if not source:
                    continue
                evidence = EvidenceCitation(
                    source_kind="transcript_segment",
                    transcript_segment_id=source.id,
                    repository_material_id=source.repository_material_id,
                    repository_segment_id=source.repository_segment_id,
                    source_locator=source.source_locator,
                    citation_text_original=source.original_text,
                )
                session.add(evidence)
                session.flush()
                session.add(
                    RecordEvidence(
                        record_id=record.id,
                        evidence_id=evidence.id,
                        relation="supports",
                        order_index=order_index,
                    )
                )
            record_ids.append(record.id)
        run.raw_response = raw_response
        run.candidate_record_ids = record_ids
        run.status = "completed"
        run.finished_at = datetime.now(timezone.utc)
        _audit(
            session,
            entity_type="ai_extraction_run",
            entity_id=run.id,
            action="completed",
            actor_id=context.user_id,
            changes={"candidate_record_ids": record_ids},
        )
        _commit(session)
    except Exception as exc:
        run.status = "failed"
        run.error_message = str(exc)
        run.finished_at = datetime.now(timezone.utc)
        _commit(session)
        raise HTTPException(status_code=502, detail=f"AI candidate extraction failed: {exc}") from exc
    return {
        "run_id": run.id,
        "status": run.status,
        "provider": run.provider,
        "model": run.model,
        "prompt_version": run.prompt_version,
        "prompt_hash": run.prompt_hash,
        "candidate_record_ids": run.candidate_record_ids,
    }


@router.get("/narrative-arcs", response_model=list[NarrativeArcOut])
def list_narrative_arcs(
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    return list(session.scalars(select(NarrativeArc).order_by(NarrativeArc.created_at)))


@router.post("/narrative-arcs", response_model=NarrativeArcOut, status_code=status.HTTP_201_CREATED)
def create_narrative_arc(
    payload: NarrativeArcCreate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    arc = NarrativeArc(**payload.model_dump())
    session.add(arc)
    session.flush()
    _audit(session, entity_type="narrative_arc", entity_id=arc.id, action="created", actor_id=context.user_id)
    _commit(session)
    session.refresh(arc)
    return arc


@router.get("/narrative-arcs/{arc_id}")
def narrative_arc_detail(
    arc_id: str,
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    arc = session.get(NarrativeArc, arc_id)
    if not arc:
        raise _not_found("Narrative arc")
    links = list(session.scalars(select(ArcLink).where(ArcLink.arc_id == arc_id).order_by(ArcLink.order_index)))
    return {"arc": NarrativeArcOut.model_validate(arc), "links": links}


@router.post("/narrative-arcs/{arc_id}/links", status_code=status.HTTP_201_CREATED)
def create_arc_link(
    arc_id: str,
    payload: ArcLinkCreate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    if not session.get(NarrativeArc, arc_id):
        raise _not_found("Narrative arc")
    if payload.source_kind == "research_record" and not session.get(ResearchRecord, payload.source_id):
        raise _not_found("Research record")
    if payload.source_kind == "transcript_segment" and not session.get(TranscriptSegment, payload.source_id):
        raise _not_found("Transcript segment")
    link = ArcLink(arc_id=arc_id, **payload.model_dump())
    session.add(link)
    session.flush()
    _audit(session, entity_type="arc_link", entity_id=link.id, action="created", actor_id=context.user_id)
    _commit(session, "That source is already linked to this narrative arc.")
    session.refresh(link)
    return link


@router.get("/reports/meeting/{meeting_id}", response_model=ReportPayload)
def export_meeting_report(
    meeting_id: str,
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    meeting = session.get(Meeting, meeting_id)
    if not meeting:
        raise _not_found("Meeting")
    return ReportPayload(
        filename=f"{meeting.meeting_date.strftime('%Y%m%d')}_meeting_report.md",
        media_type="text/markdown;charset=utf-8",
        content=meeting_report(session, meeting),
    )


@router.get("/reports/decisions", response_model=ReportPayload)
def export_decision_report(
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    return ReportPayload(
        filename="decision_log.md",
        media_type="text/markdown;charset=utf-8",
        content=record_report(session, {"decision"}, "Decision Log"),
    )


@router.get("/reports/framework", response_model=ReportPayload)
def export_framework_report(
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    return ReportPayload(
        filename="space_time_framework.md",
        media_type="text/markdown;charset=utf-8",
        content=record_report(
            session,
            {"spatial_issue", "temporal_issue", "uncertainty_issue", "method_claim"},
            "Space, Time, and Uncertainty Framework",
        ),
    )


@router.get("/reports/repository-enrichment", response_model=ReportPayload)
def export_repository_enrichment(
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    return ReportPayload(
        filename="repository_enrichment_queue.csv",
        media_type="text/csv;charset=utf-8",
        content=repository_enrichment_csv(session),
    )


@router.get("/reports/narrative-arcs/{arc_id}", response_model=ReportPayload)
def export_narrative_arc(
    arc_id: str,
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    arc = session.get(NarrativeArc, arc_id)
    if not arc:
        raise _not_found("Narrative arc")
    return ReportPayload(
        filename=f"narrative_arc_{arc.id}.md",
        media_type="text/markdown;charset=utf-8",
        content=narrative_arc_report(session, arc),
    )


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
