"""Validated request and response shapes for the progress API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Language = Literal["ja", "en", "mixed", "unknown"]
ReviewStatus = Literal[
    "draft", "raw", "candidate", "needs_review", "accepted", "rejected", "superseded"
]


class OrmModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class MemberCreate(BaseModel):
    name_original: str = Field(min_length=1, max_length=500)
    name_en: str | None = Field(default=None, max_length=500)
    name_ja: str | None = Field(default=None, max_length=500)
    role: str | None = Field(default=None, max_length=255)
    affiliation: str | None = Field(default=None, max_length=500)
    notes: str | None = None
    clerk_user_id: str | None = Field(default=None, max_length=255)


class MemberUpdate(BaseModel):
    name_original: str | None = Field(default=None, min_length=1, max_length=500)
    name_en: str | None = Field(default=None, max_length=500)
    name_ja: str | None = Field(default=None, max_length=500)
    role: str | None = Field(default=None, max_length=255)
    affiliation: str | None = Field(default=None, max_length=500)
    notes: str | None = None
    clerk_user_id: str | None = Field(default=None, max_length=255)


class MemberOut(MemberCreate, OrmModel):
    id: str
    created_at: datetime
    updated_at: datetime


class MeetingCreate(BaseModel):
    title_original: str = Field(min_length=1, max_length=1000)
    title_en: str | None = Field(default=None, max_length=1000)
    title_ja: str | None = Field(default=None, max_length=1000)
    meeting_date: date
    location: str | None = Field(default=None, max_length=500)
    meeting_type: str = Field(default="other", max_length=64)
    summary_en: str | None = None
    summary_ja: str | None = None
    notes: str | None = None


class MeetingUpdate(BaseModel):
    title_original: str | None = Field(default=None, min_length=1, max_length=1000)
    title_en: str | None = Field(default=None, max_length=1000)
    title_ja: str | None = Field(default=None, max_length=1000)
    meeting_date: date | None = None
    location: str | None = Field(default=None, max_length=500)
    meeting_type: str | None = Field(default=None, max_length=64)
    summary_en: str | None = None
    summary_ja: str | None = None
    notes: str | None = None


class MeetingOut(MeetingCreate, OrmModel):
    id: str
    created_at: datetime
    updated_at: datetime


class SessionCreate(BaseModel):
    title_original: str = Field(min_length=1, max_length=1000)
    title_en: str | None = None
    title_ja: str | None = None
    session_type: str = "unknown"
    start_time: str | None = None
    end_time: str | None = None
    order_index: int = Field(default=0, ge=0)
    notes: str | None = None


class PresentationCreate(BaseModel):
    session_id: str | None = None
    presenter_member_id: str | None = None
    title_original: str = Field(min_length=1, max_length=1000)
    title_en: str | None = None
    title_ja: str | None = None
    language_primary: Language = "unknown"
    start_time: str | None = None
    end_time: str | None = None
    order_index: int = Field(default=0, ge=0)
    summary_en: str | None = None
    summary_ja: str | None = None
    notes: str | None = None


class MaterialLinkCreate(BaseModel):
    repository_material_id: str = Field(min_length=1, max_length=36)
    presentation_id: str | None = None
    link_status: str = "linked"
    notes: str | None = None


class EvidenceCreate(BaseModel):
    source_kind: Literal[
        "transcript_segment",
        "repository_material",
        "repository_segment",
        "repository_image",
        "external_url",
        "note",
    ]
    transcript_segment_id: str | None = None
    repository_material_id: str | None = None
    repository_segment_id: int | None = None
    repository_image_id: str | None = None
    external_url: str | None = Field(default=None, max_length=2000)
    source_locator: str | None = Field(default=None, max_length=500)
    citation_text_original: str | None = None
    citation_text_en: str | None = None
    citation_text_ja: str | None = None
    source_hash: str | None = Field(default=None, max_length=128)
    notes: str | None = None

    @model_validator(mode="after")
    def has_source_identifier(self):
        required = {
            "transcript_segment": self.transcript_segment_id,
            "repository_material": self.repository_material_id,
            "repository_segment": self.repository_segment_id,
            "repository_image": self.repository_image_id,
            "external_url": self.external_url,
            "note": self.citation_text_original,
        }
        if required[self.source_kind] in {None, ""}:
            raise ValueError(f"{self.source_kind} evidence is missing its source identifier")
        return self


class ResearchRecordCreate(BaseModel):
    record_type: str = Field(min_length=1, max_length=64)
    title_original: str | None = Field(default=None, max_length=1000)
    title_en: str | None = Field(default=None, max_length=1000)
    title_ja: str | None = Field(default=None, max_length=1000)
    summary_en: str | None = None
    summary_ja: str | None = None
    original_evidence: str | None = None
    original_language: Language = "unknown"
    meeting_id: str | None = None
    presentation_id: str | None = None
    member_id: str | None = None
    status: ReviewStatus = "draft"
    confidence: float | None = Field(default=None, ge=0, le=1)
    details: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
    evidence: list[EvidenceCreate] = Field(default_factory=list)
    concept_ids: list[str] = Field(default_factory=list)


class ResearchRecordUpdate(BaseModel):
    title_original: str | None = Field(default=None, max_length=1000)
    title_en: str | None = Field(default=None, max_length=1000)
    title_ja: str | None = Field(default=None, max_length=1000)
    summary_en: str | None = None
    summary_ja: str | None = None
    original_evidence: str | None = None
    original_language: Language | None = None
    meeting_id: str | None = None
    presentation_id: str | None = None
    member_id: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    details: dict[str, Any] | None = None
    notes: str | None = None


class ReviewRequest(BaseModel):
    decision: Literal["accepted", "rejected", "needs_review", "superseded"]
    notes: str | None = None


class EvidenceOut(EvidenceCreate, OrmModel):
    id: str


class ResearchRecordOut(OrmModel):
    id: str
    record_type: str
    title_original: str | None
    title_en: str | None
    title_ja: str | None
    summary_en: str | None
    summary_ja: str | None
    original_evidence: str | None
    original_language: str
    meeting_id: str | None
    presentation_id: str | None
    member_id: str | None
    status: str
    confidence: float | None
    details: dict[str, Any]
    created_by: str
    reviewed_by: str | None
    reviewed_at: datetime | None
    notes: str | None
    evidence: list[EvidenceOut] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ConceptCreate(BaseModel):
    label_en: str = Field(min_length=1, max_length=500)
    label_ja: str = Field(min_length=1, max_length=500)
    aliases_en: list[str] = Field(default_factory=list)
    aliases_ja: list[str] = Field(default_factory=list)
    definition_en: str | None = None
    definition_ja: str | None = None
    concept_type: str = "other"
    source_vocabulary: str | None = None
    source_version: str | None = None


class ConceptOut(ConceptCreate, OrmModel):
    id: str
    created_at: datetime
    updated_at: datetime
