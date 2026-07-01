#!/usr/bin/env python3
"""Idempotently import one meeting archive into Repository and Progress."""

from __future__ import annotations

import argparse
import hashlib
import mimetypes
import os
import shutil
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.repository import get_connection, init_repository_db, now_iso  # noqa: E402
from app.progress.database import Base, ENGINE, SessionLocal  # noqa: E402
from app.progress.models import (  # noqa: E402
    AuditEvent,
    Meeting,
    RepositoryMaterialLink,
    TranscriptSegment,
)
from app.progress.transcripts import TranscriptCue, parse_transcript  # noqa: E402
from app.repository.settings import FILES_ROOT  # noqa: E402


SUPPORTED_EXTENSIONS = {".pdf", ".ppt", ".pptx", ".rtf", ".txt", ".md", ".csv", ".vtt", ".srt"}
TRANSCRIPT_EXTENSIONS = {".txt", ".vtt", ".srt"}
NAMESPACE = uuid.UUID("4f5ad940-ced6-4915-8155-6ba8111e165d")


def default_archive_root() -> Path:
    project_root = Path(__file__).resolve().parents[4]
    return project_root / "Data Governance" / "project_progress"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def date_folder(root: Path, meeting_date: str) -> Path:
    materials_root = root / "Presentation Materials"
    for child in materials_root.iterdir():
        if child.is_dir() and child.name.strip() == meeting_date:
            return child
    raise FileNotFoundError(f"No presentation folder matches {meeting_date}")


def meeting_files(root: Path, meeting_date: str) -> tuple[list[Path], list[Path]]:
    materials = sorted(
        path for path in date_folder(root, meeting_date).iterdir() if path.is_file() and not path.name.startswith(".")
    )
    short_date = meeting_date[2:]
    minutes_root = root / "Minutes"
    transcripts = sorted(
        path
        for path in minutes_root.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and (path.name.startswith(meeting_date) or path.name.startswith(short_date))
    )
    return materials, transcripts


def display_title(path: Path, meeting_date: str) -> str:
    stem = path.stem
    for prefix in (meeting_date, meeting_date[2:]):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :].lstrip("_ -")
            break
    return stem or path.stem


def source_type(path: Path, is_transcript: bool) -> str:
    if is_transcript:
        return "workshop_material"
    if path.suffix.lower() in {".ppt", ".pptx", ".ppsx", ".pptm"}:
        return "presentation"
    if path.suffix.lower() == ".pdf":
        return "pdf"
    return "other"


def parser_state(path: Path) -> tuple[str, str | None]:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return "pending", None
    return "preserved_only", f"No structured parser is configured for {suffix or 'this file type'}."


def ensure_meeting(meeting_date: str) -> Meeting:
    parsed_date = date.fromisoformat(f"{meeting_date[:4]}-{meeting_date[4:6]}-{meeting_date[6:]}")
    with SessionLocal() as session:
        meeting = session.query(Meeting).filter(Meeting.meeting_date == parsed_date).one_or_none()
        if meeting:
            return meeting
        meeting = Meeting(
            title_original=f"国際先導研究会 {parsed_date.isoformat()}",
            title_en=f"Spatiotemporal Linguistics Research Meeting {parsed_date.isoformat()}",
            meeting_date=parsed_date,
            meeting_type="hybrid",
        )
        session.add(meeting)
        session.flush()
        session.add(
            AuditEvent(
                entity_type="meeting",
                entity_id=meeting.id,
                action="archive_imported",
                actor_id="archive-importer",
                changes={"meeting_date": meeting_date},
            )
        )
        session.commit()
        session.refresh(meeting)
        return meeting


def import_repository_file(
    path: Path,
    *,
    archive_root: Path,
    meeting_date: str,
    is_transcript: bool,
) -> tuple[str, str, bool]:
    sha256 = file_sha256(path)
    with get_connection() as connection:
        existing = connection.execute(
            "SELECT id, material_id FROM files WHERE sha256 = ? LIMIT 1", (sha256,)
        ).fetchone()
        if existing:
            return existing["material_id"], existing["id"], False

        material_id = str(uuid.uuid5(NAMESPACE, f"material:{sha256}"))
        file_id = str(uuid.uuid5(NAMESPACE, f"file:{sha256}"))
        timestamp = now_iso()
        status_value, parser_message = parser_state(path)
        relative_source = str(path.relative_to(archive_root))
        connection.execute(
            """
            INSERT INTO materials (
                id, title, authors, year, source_type, collection,
                abstract_or_notes, source_url, language, region,
                uploaded_by, raw_reference, keywords, auto_keywords,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                material_id,
                display_title(path, meeting_date),
                None,
                meeting_date[:4],
                source_type(path, is_transcript),
                f"project_progress/{meeting_date}",
                "Imported from the project progress archive.",
                None,
                None,
                None,
                "archive-importer",
                relative_source,
                None,
                None,
                "ready_for_text_extraction" if status_value == "pending" else "needs_review",
                timestamp,
                timestamp,
            ),
        )
        material_dir = Path(FILES_ROOT) / material_id
        material_dir.mkdir(parents=True, exist_ok=True)
        stored_path = material_dir / f"{file_id}{path.suffix.lower()}"
        shutil.copy2(path, stored_path)
        original_mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        connection.execute(
            """
            INSERT INTO files (
                id, material_id, original_filename, stored_path, mime_type,
                file_size, uploaded_at, sha256, ingest_source, original_mtime,
                parser_status, parser_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                material_id,
                path.name,
                str(stored_path),
                mimetypes.guess_type(path.name)[0],
                path.stat().st_size,
                timestamp,
                sha256,
                relative_source,
                original_mtime,
                status_value,
                parser_message,
            ),
        )
        return material_id, file_id, True


def link_material(meeting_id: str, material_id: str) -> bool:
    with SessionLocal() as session:
        existing = session.query(RepositoryMaterialLink).filter_by(
            meeting_id=meeting_id, repository_material_id=material_id
        ).one_or_none()
        if existing:
            return False
        link = RepositoryMaterialLink(
            meeting_id=meeting_id,
            repository_material_id=material_id,
            link_status="linked",
            notes="Created by the archive importer.",
        )
        session.add(link)
        session.flush()
        session.add(
            AuditEvent(
                entity_type="repository_material_link",
                entity_id=link.id,
                action="archive_imported",
                actor_id="archive-importer",
                changes={"repository_material_id": material_id},
            )
        )
        session.commit()
        return True


def decode_transcript(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "shift_jis"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def repository_transcript_segments(
    material_id: str,
    file_id: str,
    filename: str,
    cues: list[TranscriptCue],
) -> dict[str, int]:
    segment_ids: dict[str, int] = {}
    with get_connection() as connection:
        existing = connection.execute(
            "SELECT id, page_ref FROM extracted_segments WHERE file_id = ?", (file_id,)
        ).fetchall()
        if existing:
            return {row["page_ref"]: row["id"] for row in existing}
        timestamp = now_iso()
        for cue in cues:
            cursor = connection.execute(
                """
                INSERT INTO extracted_segments (
                    material_id, file_id, source_kind, source_locator,
                    page_ref, page_index, content_text, char_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    material_id,
                    file_id,
                    "file_transcript",
                    filename,
                    cue.source_locator,
                    cue.index,
                    cue.text,
                    len(cue.text),
                    timestamp,
                ),
            )
            segment_ids[cue.source_locator] = int(cursor.lastrowid)
    return segment_ids


def progress_transcript_segments(
    meeting_id: str,
    material_id: str,
    cues: list[TranscriptCue],
    repository_segment_ids: dict[str, int],
) -> int:
    created = 0
    with SessionLocal() as session:
        for cue in cues:
            exists = session.query(TranscriptSegment).filter_by(
                repository_material_id=material_id,
                source_locator=cue.source_locator,
            ).one_or_none()
            if exists:
                continue
            session.add(
                TranscriptSegment(
                    meeting_id=meeting_id,
                    speaker_text=cue.speaker,
                    start_seconds=cue.start_seconds,
                    end_seconds=cue.end_seconds,
                    original_text=cue.text,
                    language=cue.language,
                    repository_material_id=material_id,
                    repository_segment_id=repository_segment_ids.get(cue.source_locator),
                    source_locator=cue.source_locator,
                )
            )
            created += 1
        session.commit()
    return created


def run_import(archive_root: Path, meeting_date: str, dry_run: bool) -> dict[str, int]:
    materials, transcripts = meeting_files(archive_root, meeting_date)
    all_files = [(path, False) for path in materials] + [(path, True) for path in transcripts]
    summary = {
        "materials_found": len(materials),
        "transcripts_found": len(transcripts),
        "repository_files_created": 0,
        "material_links_created": 0,
        "transcript_segments_created": 0,
        "preserved_only": sum(1 for path, _ in all_files if path.suffix.lower() not in SUPPORTED_EXTENSIONS),
    }
    if dry_run:
        return summary

    init_repository_db()
    Base.metadata.create_all(ENGINE)
    meeting = ensure_meeting(meeting_date)
    for path, is_transcript in all_files:
        material_id, file_id, created = import_repository_file(
            path,
            archive_root=archive_root,
            meeting_date=meeting_date,
            is_transcript=is_transcript,
        )
        summary["repository_files_created"] += int(created)
        summary["material_links_created"] += int(link_material(meeting.id, material_id))
        if is_transcript and path.suffix.lower() in TRANSCRIPT_EXTENSIONS:
            cues = parse_transcript(decode_transcript(path), path.name)
            repository_segments = repository_transcript_segments(material_id, file_id, path.name, cues)
            summary["transcript_segments_created"] += progress_transcript_segments(
                meeting.id, material_id, cues, repository_segments
            )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting-date", default="20250427", help="Meeting date in YYYYMMDD format.")
    parser.add_argument("--archive-root", type=Path, default=default_archive_root())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    archive_root = args.archive_root.expanduser().resolve()
    if not archive_root.is_dir():
        parser.error(f"Archive root does not exist: {archive_root}")
    summary = run_import(archive_root, args.meeting_date, args.dry_run)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
