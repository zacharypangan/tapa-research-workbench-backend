"""Evidence-linked Research Progress Portal API."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload
from app.repository.jobs import enqueue_repository_job

from app.auth import AuthContext, require_permission
from app.api.repository import get_connection as get_repository_connection
from app.progress.archive_structure import (
    LinkedMaterialSnapshot,
    canonical_material_title,
    generated_presentation,
    generated_session,
    is_transcript_snapshot,
    normalized_title_key,
)
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
    MaterialLinkOut,
    MaterialLinkUpdate,
    MeetingCreate,
    MeetingOut,
    MeetingUpdate,
    MemberCreate,
    MemberOut,
    MemberUpdate,
    PresentationCreate,
    PresentationOut,
    PresentationUpdate,
    ResearchRecordCreate,
    ResearchRecordOut,
    ResearchRecordUpdate,
    ReviewRequest,
    AlignmentCreate,
    AlignmentOut,
    SearchResult,
    SessionCreate,
    SessionOut,
    SessionUpdate,
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


def _meeting_date_token(meeting: Meeting) -> str:
    return meeting.meeting_date.strftime("%Y%m%d")


def _record_title(record: ResearchRecord) -> str:
    return record.title_original or record.title_en or record.title_ja or "Untitled record"


def _arc_link_payload(
    session: Session,
    link: ArcLink,
    *,
    current_meeting_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": link.id,
        "source_kind": link.source_kind,
        "source_id": link.source_id,
        "role_in_arc": link.role_in_arc,
        "summary_en": link.summary_en,
        "summary_ja": link.summary_ja,
        "confidence": link.confidence,
        "order_index": link.order_index,
        "notes": link.notes,
        "meeting_id": None,
        "meeting_date": None,
        "source_label": link.source_id,
        "source_status": None,
        "source_record_type": None,
        "is_current_meeting": False,
    }
    if link.source_kind == "research_record":
        record = session.get(ResearchRecord, link.source_id)
        if record:
            payload.update(
                {
                    "meeting_id": record.meeting_id,
                    "source_label": _record_title(record),
                    "source_status": record.status,
                    "source_record_type": record.record_type,
                }
            )
            if record.meeting_id:
                meeting = session.get(Meeting, record.meeting_id)
                payload["meeting_date"] = meeting.meeting_date.isoformat() if meeting else None
    elif link.source_kind == "transcript_segment":
        segment = session.get(TranscriptSegment, link.source_id)
        if segment:
            preview = segment.original_text[:160]
            payload.update(
                {
                    "meeting_id": segment.meeting_id,
                    "source_label": f"{segment.speaker_text or 'Unknown speaker'}: {preview}",
                }
            )
            meeting = session.get(Meeting, segment.meeting_id)
            payload["meeting_date"] = meeting.meeting_date.isoformat() if meeting else None
    payload["is_current_meeting"] = bool(
        current_meeting_id and payload["meeting_id"] == current_meeting_id
    )
    return payload


def _arc_detail_payload(
    session: Session,
    arc: NarrativeArc,
    *,
    current_meeting_id: str | None = None,
) -> dict[str, Any]:
    links = list(
        session.scalars(select(ArcLink).where(ArcLink.arc_id == arc.id).order_by(ArcLink.order_index))
    )
    link_payload = [
        _arc_link_payload(session, link, current_meeting_id=current_meeting_id)
        for link in links
    ]
    current_meeting_steps = sum(1 for item in link_payload if item["is_current_meeting"])
    return {
        "arc": NarrativeArcOut.model_validate(arc),
        "links": link_payload,
        "link_count": len(link_payload),
        "current_meeting_steps": current_meeting_steps,
    }


def _narrative_treatment() -> list[dict[str, str]]:
    return [
        {
            "title": "Evidence-linked storytelling",
            "description": (
                "Narrative arcs do not float above the ledger. Each arc step is anchored in a "
                "research record or transcript segment so the story remains inspectable."
            ),
        },
        {
            "title": "Chronology before rhetoric",
            "description": (
                "Arc development is shown across meetings in date order, making it clear how ideas "
                "move from planning, to pilot validation, to bilingual retrieval, to release hardening."
            ),
        },
        {
            "title": "Reviewer-governed interpretation",
            "description": (
                "The portal treats narrative as a governed interpretation layer. Accepted records, "
                "confidence notes, and provenance metadata remain visible even when the story is condensed."
            ),
        },
        {
            "title": "Original-first multilingual context",
            "description": (
                "Narrative summaries can be bilingual, but the underlying evidence chain keeps original "
                "language text and exact source locators available for scholarly review."
            ),
        },
    ]


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


@router.get("/dashboard")
def progress_dashboard(
    session: Session = Depends(get_session),
    _context: AuthContext = Depends(require_permission("progress:read")),
):
    meetings = list(session.scalars(select(Meeting).order_by(Meeting.meeting_date)))
    sessions = list(session.scalars(select(MeetingSession)))
    presentations = list(session.scalars(select(Presentation)))
    materials = list(session.scalars(select(RepositoryMaterialLink)))
    records = list(
        session.scalars(
            select(ResearchRecord)
            .options(selectinload(ResearchRecord.evidence_links).selectinload(RecordEvidence.evidence))
            .order_by(ResearchRecord.created_at)
        )
    )
    arcs = [
        _arc_detail_payload(session, arc)
        for arc in session.scalars(select(NarrativeArc).order_by(NarrativeArc.created_at))
    ]
    transcript_counts = {
        meeting_id: count
        for meeting_id, count in session.execute(
            select(TranscriptSegment.meeting_id, func.count())
            .group_by(TranscriptSegment.meeting_id)
        )
    }
    session_counts = Counter(item.meeting_id for item in sessions)
    presentation_counts = Counter(item.meeting_id for item in presentations)
    material_counts = Counter(item.meeting_id for item in materials)
    record_counts = Counter(item.meeting_id for item in records if item.meeting_id)
    accepted_record_counts = Counter(
        item.meeting_id for item in records if item.meeting_id and item.status == "accepted"
    )
    featured_titles_by_meeting: dict[str, list[str]] = {}
    for meeting in meetings:
        featured_titles_by_meeting[meeting.id] = [
            _record_title(record)
            for record in records
            if record.meeting_id == meeting.id
        ][:3]

    arc_steps_by_meeting = Counter()
    arc_titles_by_meeting: dict[str, set[str]] = {}
    for arc in arcs:
        for link in arc["links"]:
            meeting_id = link.get("meeting_id")
            if not meeting_id:
                continue
            arc_steps_by_meeting[meeting_id] += 1
            arc_titles_by_meeting.setdefault(meeting_id, set()).add(arc["arc"].title_en)

    timeline = [
        {
            "meeting": MeetingOut.model_validate(meeting),
            "session_count": session_counts.get(meeting.id, 0),
            "presentation_count": presentation_counts.get(meeting.id, 0),
            "material_count": material_counts.get(meeting.id, 0),
            "transcript_count": transcript_counts.get(meeting.id, 0),
            "record_count": record_counts.get(meeting.id, 0),
            "accepted_record_count": accepted_record_counts.get(meeting.id, 0),
            "narrative_arc_step_count": arc_steps_by_meeting.get(meeting.id, 0),
            "featured_record_titles": featured_titles_by_meeting.get(meeting.id, []),
            "arc_titles": sorted(arc_titles_by_meeting.get(meeting.id, set())),
        }
        for meeting in meetings
    ]
    return {
        "summary": {
            "meeting_count": len(meetings),
            "session_count": len(sessions),
            "presentation_count": len(presentations),
            "material_count": len(materials),
            "record_count": len(records),
            "accepted_record_count": sum(1 for record in records if record.status == "accepted"),
            "narrative_arc_count": len(arcs),
            "transcript_segment_count": sum(transcript_counts.values()),
        },
        "narrative_treatment": _narrative_treatment(),
        "timeline": timeline,
        "arcs": arcs,
    }


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
                       m.raw_reference,
                       f.original_filename, f.parser_status, f.parser_message
                FROM materials m
                LEFT JOIN files f ON f.material_id = m.id
                WHERE m.id IN ({placeholders})
                """,
                material_ids,
            ).fetchall()
        repository_metadata = {row["id"]: dict(row) for row in rows}
    meeting_date = _meeting_date_token(meeting)
    snapshots: list[LinkedMaterialSnapshot] = []
    presentations_by_id = {
        presentation.id: presentation
        for presentation in presentations
    }
    presentation_material_counts = Counter(
        link.presentation_id for link in materials if link.presentation_id
    )
    session_presentation_counts = Counter(
        presentation.session_id for presentation in presentations if presentation.session_id
    )
    session_material_counts = Counter()
    for link in materials:
        if not link.presentation_id:
            continue
        presentation = presentations_by_id.get(link.presentation_id)
        if presentation and presentation.session_id:
            session_material_counts[presentation.session_id] += 1
    material_payload = []
    for link in materials:
        metadata = repository_metadata.get(link.repository_material_id, {})
        snapshot = LinkedMaterialSnapshot(
            link_id=link.id,
            repository_material_id=link.repository_material_id,
            repository_title=metadata.get("title"),
            original_filename=metadata.get("original_filename"),
            source_type=metadata.get("source_type"),
            raw_reference=metadata.get("raw_reference"),
            parser_status=metadata.get("parser_status"),
        )
        snapshots.append(snapshot)
        linked_presentation = presentations_by_id.get(link.presentation_id) if link.presentation_id else None
        material_payload.append(
            {
                "id": link.id,
                "repository_material_id": link.repository_material_id,
                "presentation_id": link.presentation_id,
                "link_status": link.link_status,
                "notes": link.notes,
                "repository_title": metadata.get("title"),
                "original_filename": metadata.get("original_filename"),
                "source_type": metadata.get("source_type"),
                "repository_status": metadata.get("status"),
                "parser_status": metadata.get("parser_status"),
                "parser_message": metadata.get("parser_message"),
                "raw_reference": metadata.get("raw_reference"),
                "display_title": canonical_material_title(
                    metadata.get("title"),
                    original_filename=metadata.get("original_filename"),
                    meeting_date=meeting_date,
                ),
                "group_key": normalized_title_key(
                    metadata.get("title"),
                    original_filename=metadata.get("original_filename"),
                    meeting_date=meeting_date,
                ),
                "is_transcript_source": is_transcript_snapshot(snapshot),
                "presentation_title": (
                    linked_presentation.title_original
                    if linked_presentation
                    else None
                ),
                "generated_presentation": generated_presentation(linked_presentation.notes if linked_presentation else None),
            }
        )
    transcript_material_count = sum(1 for snapshot in snapshots if is_transcript_snapshot(snapshot))
    generated_session_count = sum(1 for item in sessions if generated_session(item.notes))
    generated_presentation_count = sum(1 for item in presentations if generated_presentation(item.notes))
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
    related_record_ids = {record.id for record in records}
    related_segment_ids = {segment.id for segment in transcript_segments}
    arc_ids: set[str] = set()
    if related_record_ids:
        arc_ids.update(
            session.scalars(
                select(ArcLink.arc_id).where(
                    ArcLink.source_kind == "research_record",
                    ArcLink.source_id.in_(related_record_ids),
                )
            )
        )
    if related_segment_ids:
        arc_ids.update(
            session.scalars(
                select(ArcLink.arc_id).where(
                    ArcLink.source_kind == "transcript_segment",
                    ArcLink.source_id.in_(related_segment_ids),
                )
            )
        )
    related_arcs = [
        _arc_detail_payload(session, arc, current_meeting_id=meeting_id)
        for arc in session.scalars(
            select(NarrativeArc).where(NarrativeArc.id.in_(sorted(arc_ids))).order_by(NarrativeArc.created_at)
        )
    ] if arc_ids else []
    return {
        "meeting": MeetingOut.model_validate(meeting),
        "structure": {
            "session_count": len(sessions),
            "presentation_count": len(presentations),
            "material_count": len(materials),
            "presentation_material_count": len(materials) - transcript_material_count,
            "transcript_material_count": transcript_material_count,
            "has_presentations": bool(presentations),
            "transcript_only": transcript_material_count > 0 and len(materials) == transcript_material_count,
            "generated_session_count": generated_session_count,
            "generated_presentation_count": generated_presentation_count,
            "related_arc_count": len(related_arcs),
        },
        "sessions": [
            {
                **SessionOut.model_validate(item).model_dump(),
                "generated": generated_session(item.notes),
                "presentation_count": session_presentation_counts.get(item.id, 0),
                "material_count": session_material_counts.get(item.id, 0),
            }
            for item in sessions
        ],
        "presentations": [
            {
                **PresentationOut.model_validate(item).model_dump(),
                "generated": generated_presentation(item.notes),
                "material_count": presentation_material_counts.get(item.id, 0),
            }
            for item in presentations
        ],
        "materials": material_payload,
        "transcript_segments": transcript_segments,
        "transcript_count": transcript_count,
        "records": [_record_out(record) for record in records],
        "related_arcs": related_arcs,
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


@router.patch("/sessions/{session_id}", response_model=SessionOut)
def update_session(
    session_id: str,
    payload: SessionUpdate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    meeting_session = session.get(MeetingSession, session_id)
    if not meeting_session:
        raise _not_found("Meeting session")
    changes = _apply_updates(meeting_session, payload.model_dump(exclude_unset=True))
    _audit(
        session,
        entity_type="meeting_session",
        entity_id=meeting_session.id,
        action="updated",
        actor_id=context.user_id,
        changes=changes,
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


@router.patch("/presentations/{presentation_id}", response_model=PresentationOut)
def update_presentation(
    presentation_id: str,
    payload: PresentationUpdate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    presentation = session.get(Presentation, presentation_id)
    if not presentation:
        raise _not_found("Presentation")
    updates = payload.model_dump(exclude_unset=True)
    if "session_id" in updates and updates["session_id"] is not None:
        meeting_session = session.get(MeetingSession, updates["session_id"])
        if not meeting_session or meeting_session.meeting_id != presentation.meeting_id:
            raise HTTPException(status_code=422, detail="Session does not belong to the same meeting.")
    if "presenter_member_id" in updates and updates["presenter_member_id"] is not None:
        if not session.get(Member, updates["presenter_member_id"]):
            raise HTTPException(status_code=422, detail="Presenter member was not found.")
    changes = _apply_updates(presentation, updates)
    _audit(
        session,
        entity_type="presentation",
        entity_id=presentation.id,
        action="updated",
        actor_id=context.user_id,
        changes=changes,
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


@router.patch("/material-links/{link_id}", response_model=MaterialLinkOut)
def update_material_link(
    link_id: str,
    payload: MaterialLinkUpdate,
    session: Session = Depends(get_session),
    context: AuthContext = Depends(require_permission("progress:write")),
):
    link = session.get(RepositoryMaterialLink, link_id)
    if not link:
        raise _not_found("Material link")
    updates = payload.model_dump(exclude_unset=True)
    if "presentation_id" in updates and updates["presentation_id"] is not None:
        presentation = session.get(Presentation, updates["presentation_id"])
        if not presentation or presentation.meeting_id != link.meeting_id:
            raise HTTPException(status_code=422, detail="Presentation does not belong to the same meeting.")
    changes = _apply_updates(link, updates)
    _audit(
        session,
        entity_type="repository_material_link",
        entity_id=link.id,
        action="updated",
        actor_id=context.user_id,
        changes=changes,
    )
    _commit(session)
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
    if payload.status in {"accepted", "rejected", "superseded"} and "progress:review" not in context.permissions:
        raise HTTPException(status_code=403, detail="Only reviewers may create reviewed records.")
    if payload.status == "accepted" and not payload.evidence:
        raise HTTPException(status_code=422, detail="Accepted records require evidence.")
    record_data = payload.model_dump(exclude={"evidence", "concept_ids"})
    record = ResearchRecord(**record_data, created_by=context.user_id)
    if payload.status in {"accepted", "rejected", "superseded"}:
        record.reviewed_by = context.user_id
        record.reviewed_at = datetime.now(timezone.utc)
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
    return _arc_detail_payload(session, arc)


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
    return await enqueue_repository_job(session, "repository_extract", material_id, {"force": force}, context.user_id)
