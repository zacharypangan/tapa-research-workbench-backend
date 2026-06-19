from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field


class MaterialCreate(BaseModel):
    title: str = Field(default="Untitled material", max_length=500)
    authors: Optional[str] = Field(default=None, max_length=1000)
    year: Optional[str] = Field(default=None, max_length=64)
    source_type: str = "other"
    collection: Optional[str] = Field(default=None, max_length=255)
    abstract_or_notes: Optional[str] = None
    source_url: Optional[str] = Field(default=None, max_length=2000)
    language: Optional[str] = Field(default=None, max_length=255)
    region: Optional[str] = Field(default=None, max_length=255)
    uploaded_by: Optional[str] = Field(default=None, max_length=255)
    raw_reference: Optional[str] = None
    keywords: Optional[str] = None
    auto_keywords: Optional[str] = None
    status: str = "needs_metadata"


class MaterialUpdate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=500)
    authors: Optional[str] = Field(default=None, max_length=1000)
    year: Optional[str] = Field(default=None, max_length=64)
    source_type: Optional[str] = None
    collection: Optional[str] = Field(default=None, max_length=255)
    abstract_or_notes: Optional[str] = None
    source_url: Optional[str] = Field(default=None, max_length=2000)
    language: Optional[str] = Field(default=None, max_length=255)
    region: Optional[str] = Field(default=None, max_length=255)
    uploaded_by: Optional[str] = Field(default=None, max_length=255)
    raw_reference: Optional[str] = None
    keywords: Optional[str] = None
    auto_keywords: Optional[str] = None
    status: Optional[str] = None


class ExtractRequest(BaseModel):
    include_links: bool = True
    max_link_depth: int = Field(default=1, ge=0, le=3)
    max_link_pages: int = Field(default=20, ge=1, le=200)
    max_segments: int = Field(default=1000, ge=1, le=5000)


class SearchReportRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    context_window: int = Field(default=1, ge=0, le=3)
    max_results: int = Field(default=200, ge=1, le=1000)
    material_id: Optional[str] = None


class ObservationCreate(BaseModel):
    observation_type: str = "term"
    observed_text: str = Field(..., min_length=1, max_length=1000)
    source_segment_id: Optional[int] = None
    source_image_id: Optional[str] = None
    source_page_ref: Optional[str] = Field(default=None, max_length=255)
    source_locator: Optional[str] = Field(default=None, max_length=2000)
    context_quote: Optional[str] = None
    notes: Optional[str] = None
    observed_by: Optional[str] = Field(default=None, max_length=255)


class ObservationUpdate(BaseModel):
    observation_type: Optional[str] = None
    observed_text: Optional[str] = Field(default=None, min_length=1, max_length=1000)
    source_segment_id: Optional[int] = None
    source_image_id: Optional[str] = None
    source_page_ref: Optional[str] = Field(default=None, max_length=255)
    source_locator: Optional[str] = Field(default=None, max_length=2000)
    context_quote: Optional[str] = None
    notes: Optional[str] = None
    observed_by: Optional[str] = Field(default=None, max_length=255)


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    material_id: Optional[str] = None
    limit: int = Field(default=12, ge=1, le=50)
    include_observations: bool = True
    auto_index: bool = False


class AskCorpusRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    material_id: Optional[str] = None
    max_results: int = Field(default=8, ge=1, le=20)


class BuildSemanticIndexRequest(BaseModel):
    material_id: Optional[str] = None
    limit: int = Field(default=200, ge=1, le=1000)
    force: bool = False


class MultimodalSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    material_id: Optional[str] = None
    limit: int = Field(default=12, ge=1, le=50)
    include_observations: bool = True
    include_images: bool = True
    auto_index_images: bool = False
    image_index_limit: int = Field(default=0, ge=0, le=100)


class GraphBuildRequest(BaseModel):
    material_id: Optional[str] = None
    force: bool = True


class GraphEdgeReviewRequest(BaseModel):
    review_status: str = Field(..., pattern="^(accepted|rejected|needs_review)$")


class SemanticRelationRuleResponse(BaseModel):
    rule_id: str
    predicate: str
    relation_family: str
    subject_type: str
    object_type: str
    trigger_condition: str
    required_evidence: str
    evidence_source_tables: list[str]
    confidence_logic: str
    default_status: str
    extraction_method: str
    visible_by_default: bool
    candidate_only: bool
    methodological_note: str
    limitation_caution: str
    meaning: str
    does_not_mean: str
    accepted_when: str
    needs_review_when: str
    implemented: bool


class SemanticRelationRulesResponse(BaseModel):
    policy_version: str
    rules: list[SemanticRelationRuleResponse]
    count: int
    policy: dict[str, str]


class SemanticRelationEntityResponse(BaseModel):
    id: str
    label: str
    type: str


class SemanticRelationExplanationResponse(BaseModel):
    relation_id: str
    predicate: str
    rule_id: str
    matching_rule_ids: list[str]
    plain_language_explanation: str
    subject: SemanticRelationEntityResponse
    object: SemanticRelationEntityResponse
    evidence_count: int
    document_count: int
    confidence: float
    status: str
    extraction_method: str
    evidence_items: list[dict[str, Any]]
    methodological_caution: str
    what_it_does_not_mean: str
    recommended_review_action: str
    rule: SemanticRelationRuleResponse
    properties: dict[str, Any]
    review_decision: Optional[dict[str, Any]] = None


class PlaceResolutionReviewRequest(BaseModel):
    resolution_status: str = Field(..., pattern="^(resolved|ambiguous|unresolved|rejected)$")
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    resolved_label: Optional[str] = Field(default=None, max_length=300)
    gazetteer_id: Optional[str] = Field(default=None, max_length=300)
    notes: Optional[str] = Field(default=None, max_length=2000)


class TimeResolutionReviewRequest(BaseModel):
    resolution_status: str = Field(..., pattern="^(resolved|ambiguous|invalid|needs_review|rejected)$")
    start_year: Optional[int] = Field(default=None, ge=-100000, le=100000)
    end_year: Optional[int] = Field(default=None, ge=-100000, le=100000)
    resolved_label: Optional[str] = Field(default=None, max_length=300)
    notes: Optional[str] = Field(default=None, max_length=2000)


@dataclass
class SegmentInput:
    source_kind: str
    source_locator: str
    page_ref: str
    page_index: int
    content_text: str


@dataclass
class ExtractionResult:
    segments: list[SegmentInput]
    warnings: list[str]


@dataclass
class ImageEvidenceInput:
    file_id: Optional[str]
    evidence_type: str
    source_kind: str
    source_locator: str
    page_ref: str
    page_index: int
    image_path: str
    mime_type: str
    width: int
    height: int
    extraction_method: str
    ocr_text: str = ""
    visual_caption: str = ""
    fingerprint: str = ""
