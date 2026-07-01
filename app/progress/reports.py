"""Deterministic Markdown and CSV exports from reviewed portal records."""

from __future__ import annotations

import csv
import io
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.progress.models import (
    ArcLink,
    Meeting,
    NarrativeArc,
    RecordEvidence,
    ResearchRecord,
    TranscriptSegment,
)


def title(record: ResearchRecord) -> str:
    return record.title_original or record.title_en or record.title_ja or "Untitled record"


def evidence_lines(record: ResearchRecord) -> list[str]:
    lines: list[str] = []
    for link in record.evidence_links:
        evidence = link.evidence
        locator = evidence.source_locator or evidence.external_url or evidence.id
        quote = evidence.citation_text_original or evidence.citation_text_en or evidence.citation_text_ja
        lines.append(f"  - Evidence: `{evidence.source_kind}` {locator}" + (f" — {quote}" if quote else ""))
    return lines


def meeting_report(session: Session, meeting: Meeting) -> str:
    records = list(
        session.scalars(
            select(ResearchRecord)
            .where(ResearchRecord.meeting_id == meeting.id)
            .options(selectinload(ResearchRecord.evidence_links).selectinload(RecordEvidence.evidence))
            .order_by(ResearchRecord.record_type, ResearchRecord.created_at)
        )
    )
    grouped: dict[str, list[ResearchRecord]] = defaultdict(list)
    for record in records:
        grouped[record.record_type].append(record)
    lines = [
        f"# {meeting.title_original}",
        "",
        f"- Date: {meeting.meeting_date.isoformat()}",
        f"- Location: {meeting.location or 'Not recorded'}",
        f"- Type: {meeting.meeting_type}",
        "",
        "## English Summary",
        "",
        meeting.summary_en or "Not yet reviewed.",
        "",
        "## 日本語要約",
        "",
        meeting.summary_ja or "未確認。",
    ]
    for record_type, items in grouped.items():
        lines.extend(["", f"## {record_type.replace('_', ' ').title()}", ""])
        for record in items:
            lines.append(f"- **{title(record)}** [{record.status}]")
            if record.summary_en:
                lines.append(f"  - EN: {record.summary_en}")
            if record.summary_ja:
                lines.append(f"  - JA: {record.summary_ja}")
            lines.extend(evidence_lines(record))
    return "\n".join(lines).rstrip() + "\n"


def record_report(session: Session, record_types: set[str], heading: str) -> str:
    records = list(
        session.scalars(
            select(ResearchRecord)
            .where(ResearchRecord.record_type.in_(record_types))
            .options(selectinload(ResearchRecord.evidence_links).selectinload(RecordEvidence.evidence))
            .order_by(ResearchRecord.created_at)
        )
    )
    lines = [f"# {heading}", ""]
    for record in records:
        lines.append(f"## {title(record)}")
        lines.append("")
        lines.append(f"- Type: {record.record_type}")
        lines.append(f"- Status: {record.status}")
        if record.summary_en:
            lines.append(f"- EN: {record.summary_en}")
        if record.summary_ja:
            lines.append(f"- JA: {record.summary_ja}")
        lines.extend(evidence_lines(record))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def repository_enrichment_csv(session: Session) -> str:
    records = list(
        session.scalars(
            select(ResearchRecord)
            .where(ResearchRecord.record_type == "repository_enrichment")
            .options(selectinload(ResearchRecord.evidence_links).selectinload(RecordEvidence.evidence))
            .order_by(ResearchRecord.created_at)
        )
    )
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["record_id", "title", "status", "meeting_id", "summary_en", "summary_ja", "evidence"])
    for record in records:
        locators = "; ".join(
            link.evidence.source_locator or link.evidence.external_url or link.evidence.id
            for link in record.evidence_links
        )
        writer.writerow(
            [record.id, title(record), record.status, record.meeting_id, record.summary_en, record.summary_ja, locators]
        )
    return output.getvalue()


def narrative_arc_report(session: Session, arc: NarrativeArc) -> str:
    links = list(
        session.scalars(select(ArcLink).where(ArcLink.arc_id == arc.id).order_by(ArcLink.order_index))
    )
    lines = [
        f"# {arc.title_en}",
        "",
        f"# {arc.title_ja}",
        "",
        arc.description_en,
        "",
        arc.description_ja,
        "",
        "## Chronological Evidence",
        "",
    ]
    for link in links:
        source_label = link.source_id
        if link.source_kind == "research_record":
            record = session.get(ResearchRecord, link.source_id)
            if record:
                source_label = title(record)
        elif link.source_kind == "transcript_segment":
            segment = session.get(TranscriptSegment, link.source_id)
            if segment:
                source_label = f"{segment.speaker_text or 'Unknown speaker'}: {segment.original_text[:160]}"
        lines.append(f"- **{link.role_in_arc.replace('_', ' ').title()}** — {source_label}")
        if link.summary_en:
            lines.append(f"  - EN: {link.summary_en}")
        if link.summary_ja:
            lines.append(f"  - JA: {link.summary_ja}")
    return "\n".join(lines).rstrip() + "\n"
