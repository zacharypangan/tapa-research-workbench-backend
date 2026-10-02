"""Helpers for materializing meeting/session/presentation structure from archive links."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.progress.models import AuditEvent, Meeting, MeetingSession, Presentation, RepositoryMaterialLink


SESSION_MARKER = "[archive-structure] default-session"
PRESENTATION_MARKER = "[archive-structure] generated-presentation"
TRANSCRIPT_EXTENSIONS = {".txt", ".vtt", ".srt"}
VERSION_SUFFIX_RE = re.compile(
    r"(?ix)"
    r"(?:[\s._-]+(?:v|ver|version)\.?\s*\d+(?:\.\d+)*)$"
)
BARE_VERSION_SUFFIX_RE = re.compile(r"(?ix)(?:[\s._-]+(?:ver|version|v)\.?)$")
COPY_SUFFIX_RE = re.compile(r"\s*\(\d+\)$")
DATE_PREFIX_RE = re.compile(r"^\s*(\d{8}|\d{6})[\s._-]*")
SEPARATOR_RE = re.compile(r"[\u3000\s]+")


@dataclass(frozen=True)
class LinkedMaterialSnapshot:
    link_id: str
    repository_material_id: str
    repository_title: str | None
    original_filename: str | None
    source_type: str | None
    raw_reference: str | None
    parser_status: str | None


def generated_session(notes: str | None) -> bool:
    return bool(notes and SESSION_MARKER in notes)


def generated_presentation(notes: str | None) -> bool:
    return bool(notes and PRESENTATION_MARKER in notes)


def _strip_date_prefix(value: str, meeting_date: str | None) -> str:
    if meeting_date:
        for prefix in (meeting_date, meeting_date[2:]):
            if value.startswith(prefix):
                return value[len(prefix) :].lstrip(" _-.")
    match = DATE_PREFIX_RE.match(value)
    if match:
        return value[match.end() :].lstrip(" _-.")
    return value


def canonical_material_title(
    title: str | None,
    *,
    original_filename: str | None = None,
    meeting_date: str | None = None,
) -> str:
    source = title or (Path(original_filename).stem if original_filename else "")
    normalized = unicodedata.normalize("NFKC", source).strip()
    normalized = _strip_date_prefix(normalized, meeting_date)
    normalized = re.sub(r"[_]+", " ", normalized)
    normalized = SEPARATOR_RE.sub(" ", normalized).strip(" ._-")
    return normalized or source or (original_filename or "Untitled presentation")


def normalized_title_key(
    title: str | None,
    *,
    original_filename: str | None = None,
    meeting_date: str | None = None,
) -> str:
    normalized = canonical_material_title(
        title,
        original_filename=original_filename,
        meeting_date=meeting_date,
    )
    while True:
        next_value = VERSION_SUFFIX_RE.sub("", normalized)
        next_value = BARE_VERSION_SUFFIX_RE.sub("", next_value)
        next_value = COPY_SUFFIX_RE.sub("", next_value).strip(" ._-")
        if next_value == normalized:
            break
        normalized = next_value
    normalized = re.sub(r"[-]+", " ", normalized)
    normalized = SEPARATOR_RE.sub(" ", normalized).strip(" ._-")
    return normalized.casefold() or canonical_material_title(
        title,
        original_filename=original_filename,
        meeting_date=meeting_date,
    ).casefold()


def is_transcript_snapshot(snapshot: LinkedMaterialSnapshot) -> bool:
    raw_reference = (snapshot.raw_reference or "").replace("\\", "/")
    filename = snapshot.original_filename or ""
    extension = Path(filename).suffix.lower()
    if raw_reference.startswith("Minutes/"):
        return True
    if extension in {".vtt", ".srt"}:
        return True
    return extension == ".txt" and snapshot.source_type == "workshop_material"


def _default_session_title(meeting: Meeting) -> str:
    return meeting.title_original or "Meeting materials"


def _presentation_title_from_group(group: list[LinkedMaterialSnapshot], meeting_date: str) -> str:
    titles = [
        canonical_material_title(
            item.repository_title,
            original_filename=item.original_filename,
            meeting_date=meeting_date,
        )
        for item in group
    ]
    return sorted(titles, key=lambda value: (len(value), value.casefold()))[0]


def backfill_meeting_structure(
    session: Session,
    meeting: Meeting,
    material_snapshots: list[LinkedMaterialSnapshot],
    *,
    actor_id: str = "archive-structure-backfill",
) -> dict[str, int]:
    meeting_date = meeting.meeting_date.strftime("%Y%m%d")
    meeting_sessions = list(
        session.scalars(
            select(MeetingSession)
            .where(MeetingSession.meeting_id == meeting.id)
            .order_by(MeetingSession.order_index, MeetingSession.created_at)
        )
    )
    summary = {
        "sessions_created": 0,
        "presentations_created": 0,
        "material_assignments_created": 0,
    }
    if meeting_sessions:
        default_session = meeting_sessions[0]
    else:
        default_session = MeetingSession(
            meeting_id=meeting.id,
            title_original=_default_session_title(meeting),
            title_en=meeting.title_en or "Meeting materials",
            title_ja=meeting.title_ja,
            session_type="archive_program",
            order_index=0,
            notes=SESSION_MARKER,
        )
        session.add(default_session)
        session.flush()
        session.add(
            AuditEvent(
                entity_type="meeting_session",
                entity_id=default_session.id,
                action="archive_structured",
                actor_id=actor_id,
                changes={"meeting_id": meeting.id},
            )
        )
        summary["sessions_created"] += 1

    links_by_id = {
        link.id: link
        for link in session.scalars(
            select(RepositoryMaterialLink).where(RepositoryMaterialLink.meeting_id == meeting.id)
        )
    }
    presentations = list(
        session.scalars(
            select(Presentation)
            .where(Presentation.meeting_id == meeting.id)
            .order_by(Presentation.order_index, Presentation.created_at)
        )
    )
    next_order_index = max((item.order_index for item in presentations), default=-1) + 1
    presentation_by_id = {item.id: item for item in presentations}
    presentation_by_key: dict[str, Presentation] = {}
    for item in presentations:
        key = normalized_title_key(
            item.title_original or item.title_en or item.title_ja,
            meeting_date=meeting_date,
        )
        if key and key not in presentation_by_key:
            presentation_by_key[key] = item
    linked_presentation_by_key: dict[str, Presentation] = {}
    for snapshot in material_snapshots:
        link = links_by_id.get(snapshot.link_id)
        if not link or not link.presentation_id:
            continue
        presentation = presentation_by_id.get(link.presentation_id)
        if not presentation:
            continue
        key = normalized_title_key(
            snapshot.repository_title,
            original_filename=snapshot.original_filename,
            meeting_date=meeting_date,
        )
        if key and key not in linked_presentation_by_key:
            linked_presentation_by_key[key] = presentation

    presentation_groups: dict[str, list[LinkedMaterialSnapshot]] = {}
    for snapshot in material_snapshots:
        if is_transcript_snapshot(snapshot):
            continue
        key = normalized_title_key(
            snapshot.repository_title,
            original_filename=snapshot.original_filename,
            meeting_date=meeting_date,
        )
        presentation_groups.setdefault(key, []).append(snapshot)

    for key in sorted(presentation_groups):
        group = sorted(
            presentation_groups[key],
            key=lambda item: (
                canonical_material_title(
                    item.repository_title,
                    original_filename=item.original_filename,
                    meeting_date=meeting_date,
                ).casefold(),
                (item.original_filename or "").casefold(),
                item.repository_material_id,
            ),
        )
        presentation = linked_presentation_by_key.get(key) or presentation_by_key.get(key)
        if not presentation:
            presentation = Presentation(
                meeting_id=meeting.id,
                session_id=default_session.id,
                title_original=_presentation_title_from_group(group, meeting_date),
                language_primary="unknown",
                order_index=next_order_index,
                notes=PRESENTATION_MARKER,
            )
            next_order_index += 1
            session.add(presentation)
            session.flush()
            session.add(
                AuditEvent(
                    entity_type="presentation",
                    entity_id=presentation.id,
                    action="archive_structured",
                    actor_id=actor_id,
                    changes={"meeting_id": meeting.id, "group_key": key},
                )
            )
            presentation_by_key[key] = presentation
            summary["presentations_created"] += 1
        elif not presentation.session_id:
            presentation.session_id = default_session.id

        for snapshot in group:
            link = links_by_id.get(snapshot.link_id)
            if not link or link.presentation_id:
                continue
            link.presentation_id = presentation.id
            session.add(
                AuditEvent(
                    entity_type="repository_material_link",
                    entity_id=link.id,
                    action="archive_structured",
                    actor_id=actor_id,
                    changes={"presentation_id": presentation.id},
                )
            )
            summary["material_assignments_created"] += 1

    return summary
