import hashlib
import json
import math
import re
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException


ENTITY_TYPES = {
    "material",
    "collection",
    "source_type",
    "concept",
    "place",
    "time_period",
    "agent",
}
RELATION_FAMILIES = {"structural", "bibliographic", "semantic", "spatial", "temporal"}
REVIEW_STATUSES = {"accepted", "needs_review", "rejected"}
EVIDENCE_TYPES = {"metadata", "segment", "image", "observation"}
MENTION_METHODS = {
    "metadata",
    "keyword",
    "auto_keyword",
    "exact_match",
    "observation",
    "image_ocr",
    "image_caption",
    "pattern_candidate",
    "llm_candidate",
}
CANDIDATE_METHODS = {"cooccurrence", "embedding_neighbor", "pattern_extraction", "llm_candidate"}

GENERIC_CONCEPT_LABELS = {
    "all",
    "and",
    "article",
    "chapter",
    "dan",
    "data",
    "dalam",
    "dari",
    "dengan",
    "figure",
    "image",
    "information",
    "language",
    "map",
    "material",
    "one",
    "other",
    "page",
    "paper",
    "process",
    "research",
    "section",
    "see",
    "sebagai",
    "source",
    "study",
    "table",
    "unknown",
    "untitled",
    "yang",
}

TEMPORAL_CUES = re.compile(
    r"\b(year|centur(?:y|ies)|period|dated|published|collected|recorded|from|between|during|before|after|bce|bc|ce|ad|bp)\b",
    flags=re.IGNORECASE,
)
PAGE_CUES = re.compile(
    r"(?:\bpage|\bpp?\.|\bfig(?:ure)?\.?|\btable|\bplate)\s*$",
    flags=re.IGNORECASE,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_label(value: object) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9\s.'-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def evidence_snippet(value: object, max_chars: int = 320) -> str:
    text = clean_text(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def stable_id(prefix: str, *parts: object) -> str:
    raw = "||".join(str(part or "") for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:24]}"


def parse_json(value: Optional[str], fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def compact_json(value: Optional[dict]) -> str:
    compact = {
        key: item
        for key, item in (value or {}).items()
        if item not in (None, "", [], {})
    }
    return json.dumps(compact, sort_keys=True)


def table_exists(con: sqlite3.Connection, table_name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return bool(row)


def init_semantic_graph_schema(con: sqlite3.Connection):
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_entities (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL CHECK (
                entity_type IN ('material', 'collection', 'source_type', 'concept', 'place', 'time_period', 'agent')
            ),
            label TEXT NOT NULL,
            normalized_label TEXT NOT NULL,
            canonical_key TEXT NOT NULL,
            properties_json TEXT NOT NULL,
            is_generic INTEGER NOT NULL DEFAULT 0,
            is_bridge_entity INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_mentions (
            id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            material_id TEXT NOT NULL,
            evidence_type TEXT NOT NULL CHECK (
                evidence_type IN ('metadata', 'segment', 'image', 'observation')
            ),
            segment_id INTEGER,
            image_id TEXT,
            observation_id TEXT,
            surface_text TEXT NOT NULL,
            page_ref TEXT,
            source_locator TEXT,
            snippet TEXT,
            mention_method TEXT NOT NULL CHECK (
                mention_method IN (
                    'metadata', 'keyword', 'auto_keyword', 'exact_match', 'observation',
                    'image_ocr', 'image_caption', 'pattern_candidate', 'llm_candidate'
                )
            ),
            confidence REAL NOT NULL,
            review_status TEXT NOT NULL CHECK (
                review_status IN ('accepted', 'needs_review', 'rejected')
            ),
            created_at TEXT NOT NULL,
            FOREIGN KEY(entity_id) REFERENCES kg_entities(id) ON DELETE CASCADE,
            FOREIGN KEY(material_id) REFERENCES materials(id) ON DELETE CASCADE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_relations (
            id TEXT PRIMARY KEY,
            subject_entity_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_entity_id TEXT NOT NULL,
            relation_family TEXT NOT NULL CHECK (
                relation_family IN ('structural', 'bibliographic', 'semantic', 'spatial', 'temporal')
            ),
            evidence_count INTEGER NOT NULL,
            document_count INTEGER NOT NULL,
            confidence REAL NOT NULL,
            assertion_status TEXT NOT NULL CHECK (
                assertion_status IN ('accepted', 'needs_review', 'rejected')
            ),
            extraction_method TEXT NOT NULL,
            properties_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(subject_entity_id) REFERENCES kg_entities(id) ON DELETE CASCADE,
            FOREIGN KEY(object_entity_id) REFERENCES kg_entities(id) ON DELETE CASCADE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_relation_evidence (
            id TEXT PRIMARY KEY,
            relation_id TEXT NOT NULL,
            material_id TEXT NOT NULL,
            evidence_type TEXT NOT NULL CHECK (
                evidence_type IN ('metadata', 'segment', 'image', 'observation')
            ),
            segment_id INTEGER,
            image_id TEXT,
            observation_id TEXT,
            page_ref TEXT,
            source_locator TEXT,
            snippet TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(relation_id) REFERENCES kg_relations(id) ON DELETE CASCADE,
            FOREIGN KEY(material_id) REFERENCES materials(id) ON DELETE CASCADE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_candidate_relations (
            id TEXT PRIMARY KEY,
            subject_entity_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_entity_id TEXT NOT NULL,
            candidate_method TEXT NOT NULL CHECK (
                candidate_method IN ('cooccurrence', 'embedding_neighbor', 'pattern_extraction', 'llm_candidate')
            ),
            evidence_count INTEGER NOT NULL,
            document_count INTEGER NOT NULL,
            confidence REAL NOT NULL,
            review_status TEXT NOT NULL CHECK (
                review_status IN ('accepted', 'needs_review', 'rejected')
            ),
            rationale TEXT,
            properties_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(subject_entity_id) REFERENCES kg_entities(id) ON DELETE CASCADE,
            FOREIGN KEY(object_entity_id) REFERENCES kg_entities(id) ON DELETE CASCADE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_candidate_evidence (
            id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            material_id TEXT NOT NULL,
            evidence_type TEXT NOT NULL CHECK (
                evidence_type IN ('metadata', 'segment', 'image', 'observation')
            ),
            segment_id INTEGER,
            image_id TEXT,
            observation_id TEXT,
            page_ref TEXT,
            source_locator TEXT,
            snippet TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(candidate_id) REFERENCES kg_candidate_relations(id) ON DELETE CASCADE,
            FOREIGN KEY(material_id) REFERENCES materials(id) ON DELETE CASCADE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_entity_stats (
            entity_id TEXT PRIMARY KEY,
            mention_count INTEGER NOT NULL,
            document_count INTEGER NOT NULL,
            collection_count INTEGER NOT NULL,
            degree INTEGER NOT NULL,
            accepted_relation_count INTEGER NOT NULL,
            review_relation_count INTEGER NOT NULL,
            importance_score REAL NOT NULL,
            is_generic INTEGER NOT NULL,
            is_bridge_entity INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(entity_id) REFERENCES kg_entities(id) ON DELETE CASCADE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_place_resolution (
            id TEXT PRIMARY KEY,
            mention_label TEXT NOT NULL,
            normalized_label TEXT NOT NULL UNIQUE,
            resolved_place_entity_id TEXT,
            latitude REAL,
            longitude REAL,
            gazetteer_id TEXT,
            resolution_status TEXT NOT NULL CHECK (
                resolution_status IN ('resolved', 'ambiguous', 'unresolved', 'rejected')
            ),
            confidence REAL,
            evidence_count INTEGER NOT NULL,
            notes TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(resolved_place_entity_id) REFERENCES kg_entities(id) ON DELETE SET NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_time_resolution (
            id TEXT PRIMARY KEY,
            mention_label TEXT NOT NULL,
            normalized_label TEXT NOT NULL UNIQUE,
            resolved_time_entity_id TEXT,
            time_type TEXT NOT NULL CHECK (
                time_type IN (
                    'publication_year', 'collection_year', 'historical_period',
                    'date_mention_candidate', 'invalid_page_range', 'invalid_reference_range'
                )
            ),
            start_year INTEGER,
            end_year INTEGER,
            resolution_status TEXT NOT NULL CHECK (
                resolution_status IN ('resolved', 'ambiguous', 'invalid', 'needs_review', 'rejected')
            ),
            confidence REAL,
            evidence_count INTEGER NOT NULL,
            notes TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(resolved_time_entity_id) REFERENCES kg_entities(id) ON DELETE SET NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_resolution_evidence (
            id TEXT PRIMARY KEY,
            resolution_kind TEXT NOT NULL CHECK (resolution_kind IN ('place', 'time')),
            resolution_id TEXT NOT NULL,
            material_id TEXT NOT NULL,
            evidence_type TEXT NOT NULL CHECK (
                evidence_type IN ('metadata', 'segment', 'image', 'observation')
            ),
            segment_id INTEGER,
            image_id TEXT,
            observation_id TEXT,
            page_ref TEXT,
            source_locator TEXT,
            snippet TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(material_id) REFERENCES materials(id) ON DELETE CASCADE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_review_decisions (
            id TEXT PRIMARY KEY,
            target_kind TEXT NOT NULL CHECK (
                target_kind IN ('relation', 'candidate', 'place_resolution', 'time_resolution')
            ),
            target_key TEXT NOT NULL,
            decision TEXT NOT NULL,
            notes TEXT,
            updated_at TEXT NOT NULL,
            UNIQUE(target_kind, target_key)
        )
        """
    )
    indexes = [
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_kg_entity_canonical ON kg_entities(entity_type, canonical_key)",
        "CREATE INDEX IF NOT EXISTS idx_kg_entity_type ON kg_entities(entity_type)",
        "CREATE INDEX IF NOT EXISTS idx_kg_entity_generic ON kg_entities(is_generic, is_bridge_entity)",
        "CREATE INDEX IF NOT EXISTS idx_kg_mentions_entity ON kg_mentions(entity_id)",
        "CREATE INDEX IF NOT EXISTS idx_kg_mentions_material ON kg_mentions(material_id)",
        "CREATE INDEX IF NOT EXISTS idx_kg_relations_subject ON kg_relations(subject_entity_id)",
        "CREATE INDEX IF NOT EXISTS idx_kg_relations_object ON kg_relations(object_entity_id)",
        "CREATE INDEX IF NOT EXISTS idx_kg_relations_status ON kg_relations(assertion_status, predicate)",
        "CREATE INDEX IF NOT EXISTS idx_kg_relation_evidence_relation ON kg_relation_evidence(relation_id)",
        "CREATE INDEX IF NOT EXISTS idx_kg_candidate_status ON kg_candidate_relations(review_status, candidate_method)",
        "CREATE INDEX IF NOT EXISTS idx_kg_candidate_evidence_candidate ON kg_candidate_evidence(candidate_id)",
        "CREATE INDEX IF NOT EXISTS idx_kg_resolution_evidence_lookup ON kg_resolution_evidence(resolution_kind, resolution_id)",
        "CREATE INDEX IF NOT EXISTS idx_kg_place_status ON kg_place_resolution(resolution_status)",
        "CREATE INDEX IF NOT EXISTS idx_kg_time_status ON kg_time_resolution(resolution_status)",
    ]
    for statement in indexes:
        con.execute(statement)


def entity_id(entity_type: str, canonical_key: object) -> str:
    return stable_id("kge", entity_type, normalize_label(canonical_key))


def relation_id(subject_id: str, predicate: str, object_id: str) -> str:
    return stable_id("kgr", subject_id, predicate, object_id)


def candidate_id(subject_id: str, predicate: str, object_id: str, method: str) -> str:
    ordered = sorted([subject_id, object_id]) if predicate == "co_occurs_with" else [subject_id, object_id]
    return stable_id("kgc", ordered[0], predicate, ordered[1], method)


def ensure_entity(
    con: sqlite3.Connection,
    entity_type: str,
    label: str,
    canonical_key: object,
    properties: Optional[dict] = None,
) -> str:
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"Unsupported semantic entity type: {entity_type}")
    clean_label = clean_text(label) or "Untitled"
    canonical = normalize_label(canonical_key)
    item_id = entity_id(entity_type, canonical)
    existing = con.execute(
        "SELECT properties_json, created_at FROM kg_entities WHERE id = ?",
        (item_id,),
    ).fetchone()
    merged = parse_json(existing["properties_json"], {}) if existing else {}
    merged.update({key: value for key, value in (properties or {}).items() if value not in (None, "", [], {})})
    ts = now_iso()
    con.execute(
        """
        INSERT INTO kg_entities (
            id, entity_type, label, normalized_label, canonical_key, properties_json,
            is_generic, is_bridge_entity, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            label = excluded.label,
            normalized_label = excluded.normalized_label,
            canonical_key = excluded.canonical_key,
            properties_json = excluded.properties_json,
            updated_at = excluded.updated_at
        """,
        (
            item_id,
            entity_type,
            clean_label,
            normalize_label(clean_label),
            canonical,
            compact_json(merged),
            existing["created_at"] if existing else ts,
            ts,
        ),
    )
    return item_id


def evidence_identity(evidence: dict) -> tuple:
    return (
        evidence.get("material_id"),
        evidence.get("evidence_type"),
        evidence.get("segment_id"),
        evidence.get("image_id"),
        evidence.get("observation_id"),
        evidence.get("page_ref"),
        evidence.get("source_locator"),
    )


def mention_review_key(entity_type: str, normalized: str, evidence: dict) -> tuple:
    source_id = evidence.get("segment_id") or evidence.get("image_id") or evidence.get("observation_id") or "metadata"
    return (
        evidence.get("material_id"),
        entity_type,
        normalized,
        evidence.get("evidence_type"),
        str(source_id),
    )


def add_mention(
    con: sqlite3.Connection,
    target_entity_id: str,
    entity_type: str,
    surface_text: str,
    evidence: dict,
    mention_method: str,
    confidence: float,
    review_status: str,
    legacy_mention_reviews: dict[tuple, str],
) -> dict:
    normalized = normalize_label(surface_text)
    status = legacy_mention_reviews.get(
        mention_review_key(entity_type, normalized, evidence),
        review_status,
    )
    if status not in REVIEW_STATUSES:
        status = review_status
    source_id = evidence.get("segment_id") or evidence.get("image_id") or evidence.get("observation_id") or "metadata"
    item_id = stable_id(
        "kgm",
        target_entity_id,
        evidence.get("material_id"),
        evidence.get("evidence_type"),
        source_id,
        evidence.get("page_ref"),
        normalized,
        mention_method,
    )
    item = {
        "id": item_id,
        "entity_id": target_entity_id,
        "material_id": evidence["material_id"],
        "evidence_type": evidence["evidence_type"],
        "segment_id": evidence.get("segment_id"),
        "image_id": evidence.get("image_id"),
        "observation_id": evidence.get("observation_id"),
        "surface_text": clean_text(surface_text),
        "page_ref": evidence.get("page_ref"),
        "source_locator": evidence.get("source_locator"),
        "snippet": evidence_snippet(evidence.get("snippet")),
        "mention_method": mention_method,
        "confidence": float(confidence),
        "review_status": status,
        "created_at": now_iso(),
    }
    con.execute(
        """
        INSERT INTO kg_mentions (
            id, entity_id, material_id, evidence_type, segment_id, image_id,
            observation_id, surface_text, page_ref, source_locator, snippet,
            mention_method, confidence, review_status, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            snippet = excluded.snippet,
            confidence = excluded.confidence,
            review_status = excluded.review_status
        """,
        tuple(item[key] for key in [
            "id", "entity_id", "material_id", "evidence_type", "segment_id", "image_id",
            "observation_id", "surface_text", "page_ref", "source_locator", "snippet",
            "mention_method", "confidence", "review_status", "created_at",
        ]),
    )
    return item


def split_values(value: Optional[str], max_items: int = 16) -> list[str]:
    if not value:
        return []
    results = []
    seen = set()
    for piece in re.split(r"[;\n|,]+", value):
        label = clean_text(piece).strip(" .:-")
        normalized = normalize_label(label)
        if not label or normalized in seen:
            continue
        seen.add(normalized)
        results.append(label)
        if len(results) >= max_items:
            break
    return results


def split_authors(value: Optional[str], max_items: int = 12) -> list[str]:
    if not value:
        return []
    results = []
    seen = set()
    for piece in re.split(r";|\n|\s+and\s+", value, flags=re.IGNORECASE):
        label = clean_text(piece).strip(" .:-")
        normalized = normalize_label(label)
        if not label or normalized in seen:
            continue
        seen.add(normalized)
        results.append(label)
        if len(results) >= max_items:
            break
    return results


def concept_is_high_signal(label: str) -> bool:
    normalized = normalize_label(label)
    if (
        len(normalized) < 3
        or len(label) > 120
        or normalized.isdigit()
        or "http://" in label.lower()
        or "https://" in label.lower()
    ):
        return False
    words = normalized.split()
    return 1 <= len(words) <= 8


def text_contains_label(text: Optional[str], label: str) -> bool:
    normalized_text = f" {normalize_label(text)} "
    normalized_target = normalize_label(label)
    return bool(normalized_target and f" {normalized_target} " in normalized_text)


def exact_concept_hits(text: Optional[str], labels: set[str], max_hits: int = 10) -> list[str]:
    results = []
    seen = set()
    for label in sorted(labels, key=lambda item: (-len(item), item.lower())):
        normalized = normalize_label(label)
        if normalized in seen or not concept_is_high_signal(label):
            continue
        if text_contains_label(text, label):
            seen.add(normalized)
            results.append(label)
        if len(results) >= max_hits:
            break
    return results


def explicit_coordinates(label: str, context: Optional[str] = None) -> tuple[Optional[float], Optional[float]]:
    label_match = re.fullmatch(
        r"\s*(-?\d{1,2}(?:\.\d+)?)\s*[,/]\s*(-?\d{1,3}(?:\.\d+)?)\s*",
        label or "",
    )
    cue_match = re.search(
        r"(?:lat(?:itude)?\s*[:=]?\s*)(-?\d{1,2}(?:\.\d+)?)"
        r".{0,30}?(?:lon(?:gitude)?\s*[:=]?\s*)(-?\d{1,3}(?:\.\d+)?)",
        context or "",
        flags=re.IGNORECASE,
    )
    match = cue_match or label_match
    if not match:
        return None, None
    lat = float(match.group(1))
    lon = float(match.group(2))
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon
    return None, None


def extract_place_candidates(text: Optional[str], max_items: int = 5) -> list[str]:
    if not text:
        return []
    results = []
    seen = set()
    for match in re.finditer(
        r"\b(?:in|from|near|around|at|across|within|of)\s+([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})",
        clean_text(text),
    ):
        label = match.group(1).strip(" .,:;")
        normalized = normalize_label(label)
        if (
            not normalized
            or normalized in GENERIC_CONCEPT_LABELS
            or normalized.startswith(("the ", "this ", "these "))
            or any(char.isdigit() for char in label)
            or normalized in seen
        ):
            continue
        seen.add(normalized)
        results.append(label)
        if len(results) >= max_items:
            break
    return results


def temporal_candidates(text: Optional[str], page_ref: Optional[str] = None, max_items: int = 8) -> list[dict]:
    if not text:
        return []
    current_year = datetime.now(timezone.utc).year
    cleaned = clean_text(text)
    pattern = re.compile(
        r"\b\d{3,5}\s*BP\b"
        r"|\b\d{1,4}\s*(?:BCE|BC|CE|AD)\b"
        r"|\b\d{3,4}\s*(?:-|\u2013|to)\s*\d{3,4}\b"
        r"|\b\d{3,4}\b",
        flags=re.IGNORECASE,
    )
    results = []
    seen = set()
    for match in pattern.finditer(cleaned):
        label = clean_text(match.group(0))
        normalized = normalize_label(label)
        if normalized in seen:
            continue
        seen.add(normalized)
        before = cleaned[max(0, match.start() - 28):match.start()]
        context_window = cleaned[max(0, match.start() - 50):min(len(cleaned), match.end() + 50)]
        page_hit = bool(page_ref and normalized and normalized in normalize_label(page_ref))
        page_cue = bool(PAGE_CUES.search(before))
        temporal_cue = bool(TEMPORAL_CUES.search(context_window))
        numbers = [int(value) for value in re.findall(r"\d{1,5}", label)]
        start_year: Optional[int] = None
        end_year: Optional[int] = None
        time_type = "date_mention_candidate"
        status = "needs_review"
        confidence = 0.45
        notes = "Candidate date requires review."

        if page_hit or page_cue:
            status = "invalid"
            time_type = "invalid_page_range" if len(numbers) == 2 else "invalid_reference_range"
            notes = "Value appears in a page, figure, table, or plate reference."
        elif re.search(r"\bBP\b", label, flags=re.IGNORECASE) and numbers:
            start_year = end_year = 1950 - numbers[0]
            status = "resolved"
            time_type = "historical_period"
            confidence = 0.9
            notes = "Explicit BP temporal expression."
        elif re.search(r"\b(?:BCE|BC)\b", label, flags=re.IGNORECASE) and numbers:
            start_year = end_year = -numbers[0]
            status = "resolved"
            time_type = "historical_period"
            confidence = 0.9
            notes = "Explicit BCE/BC temporal expression."
        elif re.search(r"\b(?:CE|AD)\b", label, flags=re.IGNORECASE) and numbers:
            start_year = end_year = numbers[0]
            status = "resolved"
            time_type = "historical_period"
            confidence = 0.88
            notes = "Explicit CE/AD temporal expression."
        elif len(numbers) == 2:
            start_year, end_year = sorted(numbers)
            if start_year >= 1500 and end_year <= current_year + 1:
                status = "resolved"
                time_type = "historical_period"
                confidence = 0.82
                notes = "Year range within the accepted modern range."
            elif temporal_cue:
                status = "needs_review"
                time_type = "historical_period"
                confidence = 0.6
                notes = "Older range has temporal context but requires review."
            else:
                status = "invalid"
                time_type = "invalid_page_range"
                confidence = 0.2
                notes = "Numeric range lacks explicit temporal context."
        elif numbers:
            start_year = end_year = numbers[0]
            if 1500 <= numbers[0] <= current_year + 1:
                status = "resolved"
                time_type = "historical_period"
                confidence = 0.8
                notes = "Year is within the accepted modern range."
            elif temporal_cue:
                status = "needs_review"
                time_type = "historical_period"
                confidence = 0.55
                notes = "Older numeric reference has temporal context but requires review."
            else:
                status = "invalid"
                time_type = "invalid_reference_range"
                confidence = 0.15
                notes = "Numeric reference lacks temporal context."

        results.append(
            {
                "label": label,
                "normalized_label": normalized,
                "time_type": time_type,
                "start_year": start_year,
                "end_year": end_year,
                "resolution_status": status,
                "confidence": confidence,
                "notes": notes,
            }
        )
        if len(results) >= max_items:
            break
    return results


def metadata_year_candidate(value: Optional[str]) -> Optional[dict]:
    if not value:
        return None
    candidates = temporal_candidates(value, max_items=1)
    if not candidates:
        return None
    candidate = candidates[0]
    if candidate["resolution_status"] == "resolved":
        candidate["time_type"] = "publication_year"
        candidate["confidence"] = 0.98
        candidate["notes"] = "Publication or material year from trusted metadata."
    return candidate


def source_evidence(
    material_id: str,
    evidence_type: str,
    snippet: Optional[str] = None,
    segment_id: Optional[int] = None,
    image_id: Optional[str] = None,
    observation_id: Optional[str] = None,
    page_ref: Optional[str] = None,
    source_locator: Optional[str] = None,
) -> dict:
    return {
        "material_id": material_id,
        "evidence_type": evidence_type,
        "segment_id": segment_id,
        "image_id": image_id,
        "observation_id": observation_id,
        "page_ref": page_ref,
        "source_locator": source_locator,
        "snippet": evidence_snippet(snippet),
    }


def load_review_decisions(con: sqlite3.Connection) -> dict[tuple[str, str], str]:
    rows = con.execute(
        "SELECT target_kind, target_key, decision FROM kg_review_decisions"
    ).fetchall()
    return {(row["target_kind"], row["target_key"]): row["decision"] for row in rows}


def load_resolution_overrides(con: sqlite3.Connection, table_name: str) -> dict[str, dict]:
    if not table_exists(con, table_name):
        return {}
    rows = con.execute(f"SELECT * FROM {table_name}").fetchall()
    return {
        row["normalized_label"]: dict(row)
        for row in rows
        if row["resolution_status"] in {"resolved", "rejected"}
    }


def load_legacy_review_state(con: sqlite3.Connection) -> tuple[dict[tuple, str], dict[str, str]]:
    mention_reviews: dict[tuple, str] = {}
    candidate_reviews: dict[str, str] = {}
    if not table_exists(con, "repository_graph_edges"):
        return mention_reviews, candidate_reviews
    rows = con.execute(
        """
        SELECT
            e.edge_type, e.review_status, e.extraction_method, e.evidence_ref_json,
            source.node_type AS source_type, source.label AS source_label,
            target.node_type AS target_type, target.label AS target_label
        FROM repository_graph_edges e
        JOIN repository_graph_nodes source ON source.id = e.source_node_id
        JOIN repository_graph_nodes target ON target.id = e.target_node_id
        WHERE e.review_status = 'rejected'
           OR (e.review_status = 'accepted' AND e.extraction_method IN ('cooccurrence', 'pattern_extraction'))
        """
    ).fetchall()
    for row in rows:
        evidence = parse_json(row["evidence_ref_json"], {})
        evidence_type = {
            "extracted_segment": "segment",
            "image_evidence": "image",
            "human_observation": "observation",
            "metadata": "metadata",
        }.get(evidence.get("source"), "metadata")
        evidence["evidence_type"] = evidence_type
        if row["edge_type"] == "co_occurs_with":
            source_id = entity_id("concept", row["source_label"])
            target_id = entity_id("concept", row["target_label"])
            item_id = candidate_id(source_id, "co_occurs_with", target_id, "cooccurrence")
            candidate_reviews[item_id] = row["review_status"]
            continue
        target_entity_type = {
            "concept": "concept",
            "keyword": "concept",
            "place": "place",
            "time_reference": "time_period",
        }.get(row["target_type"])
        if target_entity_type and evidence.get("material_id"):
            mention_reviews[
                mention_review_key(
                    target_entity_type,
                    normalize_label(row["target_label"]),
                    evidence,
                )
            ] = row["review_status"]
    return mention_reviews, candidate_reviews


def clear_generated_semantic_graph(con: sqlite3.Connection):
    for table_name in [
        "kg_candidate_evidence",
        "kg_candidate_relations",
        "kg_relation_evidence",
        "kg_relations",
        "kg_mentions",
        "kg_entity_stats",
        "kg_resolution_evidence",
        "kg_place_resolution",
        "kg_time_resolution",
        "kg_entities",
    ]:
        con.execute(f"DELETE FROM {table_name}")


def upsert_resolution_evidence(
    con: sqlite3.Connection,
    resolution_kind: str,
    resolution_id: str,
    evidence: dict,
):
    item_id = stable_id(
        "kgre",
        resolution_kind,
        resolution_id,
        *evidence_identity(evidence),
    )
    con.execute(
        """
        INSERT INTO kg_resolution_evidence (
            id, resolution_kind, resolution_id, material_id, evidence_type,
            segment_id, image_id, observation_id, page_ref, source_locator,
            snippet, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET snippet = excluded.snippet
        """,
        (
            item_id,
            resolution_kind,
            resolution_id,
            evidence["material_id"],
            evidence["evidence_type"],
            evidence.get("segment_id"),
            evidence.get("image_id"),
            evidence.get("observation_id"),
            evidence.get("page_ref"),
            evidence.get("source_locator"),
            evidence.get("snippet"),
            now_iso(),
        ),
    )


def upsert_place_resolution(
    con: sqlite3.Connection,
    label: str,
    evidence: dict,
    status: str,
    confidence: float,
    resolved_entity_id: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    gazetteer_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    normalized = normalize_label(label)
    item_id = stable_id("kgp", normalized)
    con.execute(
        """
        INSERT INTO kg_place_resolution (
            id, mention_label, normalized_label, resolved_place_entity_id,
            latitude, longitude, gazetteer_id, resolution_status, confidence,
            evidence_count, notes, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            resolved_place_entity_id = COALESCE(excluded.resolved_place_entity_id, kg_place_resolution.resolved_place_entity_id),
            latitude = COALESCE(excluded.latitude, kg_place_resolution.latitude),
            longitude = COALESCE(excluded.longitude, kg_place_resolution.longitude),
            gazetteer_id = COALESCE(excluded.gazetteer_id, kg_place_resolution.gazetteer_id),
            resolution_status = CASE
                WHEN kg_place_resolution.resolution_status IN ('resolved', 'rejected')
                THEN kg_place_resolution.resolution_status
                ELSE excluded.resolution_status
            END,
            confidence = MAX(COALESCE(kg_place_resolution.confidence, 0), COALESCE(excluded.confidence, 0)),
            evidence_count = kg_place_resolution.evidence_count + 1,
            notes = COALESCE(kg_place_resolution.notes, excluded.notes),
            updated_at = excluded.updated_at
        """,
        (
            item_id,
            clean_text(label),
            normalized,
            resolved_entity_id,
            latitude,
            longitude,
            gazetteer_id,
            status,
            confidence,
            notes,
            now_iso(),
        ),
    )
    upsert_resolution_evidence(con, "place", item_id, evidence)
    return item_id


def upsert_time_resolution(
    con: sqlite3.Connection,
    candidate: dict,
    evidence: dict,
    resolved_entity_id: Optional[str] = None,
) -> str:
    item_id = stable_id("kgt", candidate["normalized_label"])
    con.execute(
        """
        INSERT INTO kg_time_resolution (
            id, mention_label, normalized_label, resolved_time_entity_id,
            time_type, start_year, end_year, resolution_status, confidence,
            evidence_count, notes, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            resolved_time_entity_id = COALESCE(excluded.resolved_time_entity_id, kg_time_resolution.resolved_time_entity_id),
            time_type = excluded.time_type,
            start_year = COALESCE(excluded.start_year, kg_time_resolution.start_year),
            end_year = COALESCE(excluded.end_year, kg_time_resolution.end_year),
            resolution_status = CASE
                WHEN kg_time_resolution.resolution_status IN ('resolved', 'rejected')
                THEN kg_time_resolution.resolution_status
                ELSE excluded.resolution_status
            END,
            confidence = MAX(COALESCE(kg_time_resolution.confidence, 0), COALESCE(excluded.confidence, 0)),
            evidence_count = kg_time_resolution.evidence_count + 1,
            notes = COALESCE(kg_time_resolution.notes, excluded.notes),
            updated_at = excluded.updated_at
        """,
        (
            item_id,
            candidate["label"],
            candidate["normalized_label"],
            resolved_entity_id,
            candidate["time_type"],
            candidate.get("start_year"),
            candidate.get("end_year"),
            candidate["resolution_status"],
            candidate.get("confidence"),
            candidate.get("notes"),
            now_iso(),
        ),
    )
    upsert_resolution_evidence(con, "time", item_id, evidence)
    return item_id


def add_relation_group(
    groups: dict,
    subject_id: str,
    predicate: str,
    object_id: str,
    family: str,
    evidence: dict,
    confidence: float,
    status: str,
    extraction_method: str,
    properties: Optional[dict] = None,
):
    key = (subject_id, predicate, object_id)
    group = groups.setdefault(
        key,
        {
            "family": family,
            "evidence": {},
            "confidences": [],
            "statuses": [],
            "methods": set(),
            "properties": {},
        },
    )
    group["evidence"][evidence_identity(evidence)] = evidence
    group["confidences"].append(float(confidence))
    group["statuses"].append(status)
    group["methods"].add(extraction_method)
    group["properties"].update(properties or {})


def persist_relation_groups(
    con: sqlite3.Connection,
    groups: dict,
    review_decisions: dict[tuple[str, str], str],
):
    ts = now_iso()
    for (subject_id, predicate, object_id), group in groups.items():
        item_id = relation_id(subject_id, predicate, object_id)
        statuses = group["statuses"]
        default_status = "accepted" if "accepted" in statuses else "needs_review"
        status = review_decisions.get(("relation", item_id), default_status)
        if status not in REVIEW_STATUSES:
            status = default_status
        methods = sorted(group["methods"])
        extraction_method = methods[0] if len(methods) == 1 else "aggregated_evidence"
        evidence_items = list(group["evidence"].values())
        material_ids = {item["material_id"] for item in evidence_items}
        confidence = sum(group["confidences"]) / max(1, len(group["confidences"]))
        con.execute(
            """
            INSERT INTO kg_relations (
                id, subject_entity_id, predicate, object_entity_id, relation_family,
                evidence_count, document_count, confidence, assertion_status,
                extraction_method, properties_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                subject_id,
                predicate,
                object_id,
                group["family"],
                len(evidence_items),
                len(material_ids),
                round(confidence, 4),
                status,
                extraction_method,
                compact_json(group["properties"]),
                ts,
                ts,
            ),
        )
        for evidence in evidence_items:
            evidence_id = stable_id("kgrv", item_id, *evidence_identity(evidence))
            con.execute(
                """
                INSERT INTO kg_relation_evidence (
                    id, relation_id, material_id, evidence_type, segment_id,
                    image_id, observation_id, page_ref, source_locator, snippet, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    item_id,
                    evidence["material_id"],
                    evidence["evidence_type"],
                    evidence.get("segment_id"),
                    evidence.get("image_id"),
                    evidence.get("observation_id"),
                    evidence.get("page_ref"),
                    evidence.get("source_locator"),
                    evidence.get("snippet"),
                    ts,
                ),
            )


def compute_entity_stats(con: sqlite3.Connection) -> dict[str, dict]:
    entities = con.execute("SELECT id, entity_type, normalized_label FROM kg_entities").fetchall()
    total_materials = con.execute("SELECT COUNT(*) FROM materials").fetchone()[0] or 1
    total_collections = con.execute(
        "SELECT COUNT(DISTINCT collection) FROM materials WHERE collection IS NOT NULL AND trim(collection) != ''"
    ).fetchone()[0] or 1
    raw: dict[str, dict] = {}
    concept_degrees = []
    for entity in entities:
        stats = con.execute(
            """
            SELECT
                COUNT(*) AS mention_count,
                COUNT(DISTINCT material_id) AS document_count,
                COUNT(DISTINCT NULLIF(trim(m.collection), '')) AS collection_count,
                SUM(CASE WHEN km.mention_method = 'observation' AND km.review_status = 'accepted' THEN 1 ELSE 0 END) AS human_count
            FROM kg_mentions km
            JOIN materials m ON m.id = km.material_id
            WHERE km.entity_id = ? AND km.review_status != 'rejected'
            """,
            (entity["id"],),
        ).fetchone()
        relation_stats = con.execute(
            """
            SELECT
                COUNT(*) AS degree,
                SUM(CASE WHEN assertion_status = 'accepted' THEN 1 ELSE 0 END) AS accepted_count,
                SUM(CASE WHEN assertion_status = 'needs_review' THEN 1 ELSE 0 END) AS review_count
            FROM kg_relations
            WHERE assertion_status != 'rejected'
              AND (subject_entity_id = ? OR object_entity_id = ?)
            """,
            (entity["id"], entity["id"]),
        ).fetchone()
        candidate_count = con.execute(
            """
            SELECT COUNT(*)
            FROM kg_candidate_relations
            WHERE review_status = 'needs_review'
              AND (subject_entity_id = ? OR object_entity_id = ?)
            """,
            (entity["id"], entity["id"]),
        ).fetchone()[0]
        item = {
            "entity_type": entity["entity_type"],
            "normalized_label": entity["normalized_label"],
            "mention_count": int(stats["mention_count"] or 0),
            "document_count": int(stats["document_count"] or 0),
            "collection_count": int(stats["collection_count"] or 0),
            "human_count": int(stats["human_count"] or 0),
            "degree": int(relation_stats["degree"] or 0),
            "accepted_relation_count": int(relation_stats["accepted_count"] or 0),
            "review_relation_count": int(relation_stats["review_count"] or 0) + int(candidate_count or 0),
        }
        raw[entity["id"]] = item
        if entity["entity_type"] == "concept":
            concept_degrees.append(item["degree"])

    sorted_degrees = sorted(concept_degrees)
    degree_cutoff = sorted_degrees[max(0, math.ceil(len(sorted_degrees) * 0.95) - 1)] if sorted_degrees else 0
    con.execute("DELETE FROM kg_entity_stats")
    ts = now_iso()
    for item_id, item in raw.items():
        coverage = item["document_count"] / total_materials
        generic = (
            item["entity_type"] == "concept"
            and (
                item["normalized_label"] in GENERIC_CONCEPT_LABELS
                or (
                    coverage >= 0.8
                    and item["degree"] >= max(8, degree_cutoff)
                )
            )
        )
        document_score = min(1.0, item["document_count"] / max(2.0, total_materials * 0.4))
        collection_score = min(1.0, item["collection_count"] / max(1.0, total_collections))
        mention_score = min(1.0, math.log1p(item["mention_count"]) / math.log1p(20))
        accepted_score = min(
            1.0,
            item["accepted_relation_count"] / max(1, item["degree"]),
        )
        human_score = min(1.0, item["human_count"] / 3)
        importance = (
            0.35 * document_score
            + 0.20 * collection_score
            + 0.20 * accepted_score
            + 0.15 * human_score
            + 0.10 * mention_score
        )
        if item["document_count"] <= 1 and item["entity_type"] == "concept":
            importance *= 0.7
        if generic:
            importance *= 0.15
        bridge = (
            item["entity_type"] == "concept"
            and not generic
            and item["document_count"] >= 2
            and (item["collection_count"] >= 2 or importance >= 0.45)
        )
        con.execute(
            """
            INSERT INTO kg_entity_stats (
                entity_id, mention_count, document_count, collection_count, degree,
                accepted_relation_count, review_relation_count, importance_score,
                is_generic, is_bridge_entity, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                item["mention_count"],
                item["document_count"],
                item["collection_count"],
                item["degree"],
                item["accepted_relation_count"],
                item["review_relation_count"],
                round(importance, 4),
                int(generic),
                int(bridge),
                ts,
            ),
        )
        con.execute(
            """
            UPDATE kg_entities
            SET is_generic = ?, is_bridge_entity = ?, updated_at = ?
            WHERE id = ?
            """,
            (int(generic), int(bridge), ts, item_id),
        )
        item["is_generic"] = generic
        item["is_bridge_entity"] = bridge
        item["importance_score"] = importance
    return raw


def persist_cooccurrence_candidates(
    con: sqlite3.Connection,
    units: dict[tuple, dict],
    entity_stats: dict[str, dict],
    review_decisions: dict[tuple[str, str], str],
    legacy_candidate_reviews: dict[str, str],
):
    total_materials = con.execute("SELECT COUNT(*) FROM materials").fetchone()[0] or 1
    pair_units: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for unit in units.values():
        concept_ids = sorted(set(unit["concept_ids"]))
        for index, source_id in enumerate(concept_ids[:10]):
            for target_id in concept_ids[index + 1:10]:
                pair_units[(source_id, target_id)].append(unit["evidence"])

    ts = now_iso()
    for (source_id, target_id), evidence_items in pair_units.items():
        unique_evidence = {evidence_identity(item): item for item in evidence_items}
        items = list(unique_evidence.values())
        material_ids = {item["material_id"] for item in items}
        source_stats = entity_stats.get(source_id, {})
        target_stats = entity_stats.get(target_id, {})
        if (
            len(items) < 3
            or len(material_ids) < 2
            or source_stats.get("is_generic")
            or target_stats.get("is_generic")
            or source_stats.get("document_count", 0) / total_materials > 0.3
            or target_stats.get("document_count", 0) / total_materials > 0.3
        ):
            continue
        confidence = min(0.9, 0.45 + 0.04 * len(items) + 0.08 * len(material_ids))
        if confidence < 0.55:
            continue
        item_id = candidate_id(source_id, "co_occurs_with", target_id, "cooccurrence")
        status = review_decisions.get(
            ("candidate", item_id),
            legacy_candidate_reviews.get(item_id, "needs_review"),
        )
        if status not in REVIEW_STATUSES:
            status = "needs_review"
        con.execute(
            """
            INSERT INTO kg_candidate_relations (
                id, subject_entity_id, predicate, object_entity_id, candidate_method,
                evidence_count, document_count, confidence, review_status, rationale,
                properties_json, created_at, updated_at
            )
            VALUES (?, ?, 'co_occurs_with', ?, 'cooccurrence', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                source_id,
                target_id,
                len(items),
                len(material_ids),
                round(confidence, 4),
                status,
                (
                    f"Concepts co-occur in {len(items)} independent evidence units "
                    f"across {len(material_ids)} documents."
                ),
                compact_json({"visible_by_default": False}),
                ts,
                ts,
            ),
        )
        for evidence in items:
            evidence_id = stable_id("kgcv", item_id, *evidence_identity(evidence))
            con.execute(
                """
                INSERT INTO kg_candidate_evidence (
                    id, candidate_id, material_id, evidence_type, segment_id,
                    image_id, observation_id, page_ref, source_locator, snippet, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    item_id,
                    evidence["material_id"],
                    evidence["evidence_type"],
                    evidence.get("segment_id"),
                    evidence.get("image_id"),
                    evidence.get("observation_id"),
                    evidence.get("page_ref"),
                    evidence.get("source_locator"),
                    evidence.get("snippet"),
                    ts,
                ),
            )


def promote_accepted_candidates(con: sqlite3.Connection):
    rows = con.execute(
        "SELECT * FROM kg_candidate_relations WHERE review_status = 'accepted'"
    ).fetchall()
    for row in rows:
        promoted_id = relation_id(row["subject_entity_id"], "related_to", row["object_entity_id"])
        decision = load_review_decisions(con).get(("relation", promoted_id), "accepted")
        con.execute(
            """
            INSERT INTO kg_relations (
                id, subject_entity_id, predicate, object_entity_id, relation_family,
                evidence_count, document_count, confidence, assertion_status,
                extraction_method, properties_json, created_at, updated_at
            )
            VALUES (?, ?, 'related_to', ?, 'semantic', ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                evidence_count = excluded.evidence_count,
                document_count = excluded.document_count,
                confidence = excluded.confidence,
                assertion_status = excluded.assertion_status,
                extraction_method = excluded.extraction_method,
                properties_json = excluded.properties_json,
                updated_at = excluded.updated_at
            """,
            (
                promoted_id,
                row["subject_entity_id"],
                row["object_entity_id"],
                row["evidence_count"],
                row["document_count"],
                row["confidence"],
                decision,
                f"reviewed_{row['candidate_method']}",
                compact_json(
                    {
                        "promoted_from_candidate_id": row["id"],
                        "original_predicate": row["predicate"],
                    }
                ),
                now_iso(),
                now_iso(),
            ),
        )
        evidence_rows = con.execute(
            "SELECT * FROM kg_candidate_evidence WHERE candidate_id = ?",
            (row["id"],),
        ).fetchall()
        for evidence in evidence_rows:
            evidence_id = stable_id("kgrv", promoted_id, *evidence_identity(dict(evidence)))
            con.execute(
                """
                INSERT OR REPLACE INTO kg_relation_evidence (
                    id, relation_id, material_id, evidence_type, segment_id,
                    image_id, observation_id, page_ref, source_locator, snippet, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    promoted_id,
                    evidence["material_id"],
                    evidence["evidence_type"],
                    evidence["segment_id"],
                    evidence["image_id"],
                    evidence["observation_id"],
                    evidence["page_ref"],
                    evidence["source_locator"],
                    evidence["snippet"],
                    now_iso(),
                ),
            )


def build_semantic_graph(con: sqlite3.Connection, material_id: Optional[str] = None) -> dict:
    init_semantic_graph_schema(con)
    if material_id:
        row = con.execute("SELECT id FROM materials WHERE id = ?", (material_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Material not found")

    review_decisions = load_review_decisions(con)
    place_overrides = load_resolution_overrides(con, "kg_place_resolution")
    time_overrides = load_resolution_overrides(con, "kg_time_resolution")
    legacy_mention_reviews, legacy_candidate_reviews = load_legacy_review_state(con)
    clear_generated_semantic_graph(con)

    materials = con.execute("SELECT * FROM materials ORDER BY updated_at DESC").fetchall()
    relation_groups: dict = {}
    material_entities: dict[str, str] = {}
    unit_concepts: dict[tuple, dict] = {}
    accepted_concept_labels = {
        clean_text(label)
        for material in materials
        for label in split_values(material["keywords"])
        if concept_is_high_signal(label)
    }
    accepted_concept_labels.update(
        clean_text(row["observed_text"])
        for row in con.execute(
            "SELECT observed_text FROM observations WHERE trim(observed_text) != ''"
        ).fetchall()
        if concept_is_high_signal(row["observed_text"])
    )

    def register_concept(label: str, source: str) -> Optional[str]:
        if not concept_is_high_signal(label):
            return None
        return ensure_entity(
            con,
            "concept",
            label,
            normalize_label(label),
            {"source": source},
        )

    for material in materials:
        material_id_value = material["id"]
        title = material["title"] or "Untitled material"
        counts = con.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM extracted_segments WHERE material_id = ?) AS segment_count,
                (SELECT COUNT(*) FROM image_evidence WHERE material_id = ?) AS image_count,
                (SELECT COUNT(*) FROM observations WHERE material_id = ?) AS observation_count
            """,
            (material_id_value, material_id_value, material_id_value),
        ).fetchone()
        material_entity = ensure_entity(
            con,
            "material",
            title,
            material_id_value,
            {
                "material_id": material_id_value,
                "source_type": material["source_type"],
                "collection": material["collection"],
                "year": material["year"],
                "language": material["language"],
                "region": material["region"],
                "status": material["status"],
                "summary": evidence_snippet(material["abstract_or_notes"] or material["raw_reference"], 240),
                "section_count": counts["segment_count"],
                "image_count": counts["image_count"],
                "observation_count": counts["observation_count"],
            },
        )
        material_entities[material_id_value] = material_entity
        metadata_evidence = source_evidence(
            material_id_value,
            "metadata",
            material["abstract_or_notes"] or material["raw_reference"] or title,
        )
        add_mention(
            con,
            material_entity,
            "material",
            title,
            metadata_evidence,
            "metadata",
            1.0,
            "accepted",
            legacy_mention_reviews,
        )

        if material["collection"]:
            collection_entity = ensure_entity(
                con,
                "collection",
                material["collection"],
                material["collection"],
            )
            add_mention(
                con,
                collection_entity,
                "collection",
                material["collection"],
                metadata_evidence,
                "metadata",
                1.0,
                "accepted",
                legacy_mention_reviews,
            )
            add_relation_group(
                relation_groups,
                material_entity,
                "belongs_to_collection",
                collection_entity,
                "structural",
                metadata_evidence,
                1.0,
                "accepted",
                "metadata",
            )

        if material["source_type"]:
            source_type_entity = ensure_entity(
                con,
                "source_type",
                material["source_type"].replace("_", " ").title(),
                material["source_type"],
            )
            add_mention(
                con,
                source_type_entity,
                "source_type",
                material["source_type"],
                metadata_evidence,
                "metadata",
                1.0,
                "accepted",
                legacy_mention_reviews,
            )
            add_relation_group(
                relation_groups,
                material_entity,
                "has_source_type",
                source_type_entity,
                "structural",
                metadata_evidence,
                1.0,
                "accepted",
                "metadata",
            )

        for author in split_authors(material["authors"]):
            agent_entity = ensure_entity(con, "agent", author, author)
            add_mention(
                con,
                agent_entity,
                "agent",
                author,
                metadata_evidence,
                "metadata",
                0.99,
                "accepted",
                legacy_mention_reviews,
            )
            add_relation_group(
                relation_groups,
                material_entity,
                "authored_by",
                agent_entity,
                "bibliographic",
                metadata_evidence,
                0.99,
                "accepted",
                "metadata",
            )

        material_concepts: dict[str, list[dict]] = defaultdict(list)
        for keyword in split_values(material["keywords"]):
            concept_entity = register_concept(keyword, "manual_keyword")
            if not concept_entity:
                continue
            mention = add_mention(
                con,
                concept_entity,
                "concept",
                keyword,
                metadata_evidence,
                "keyword",
                0.98,
                "accepted",
                legacy_mention_reviews,
            )
            material_concepts[concept_entity].append(mention)
        for keyword in split_values(material["auto_keywords"]):
            concept_entity = register_concept(keyword, "auto_keyword")
            if not concept_entity:
                continue
            mention = add_mention(
                con,
                concept_entity,
                "concept",
                keyword,
                metadata_evidence,
                "auto_keyword",
                0.72,
                "needs_review",
                legacy_mention_reviews,
            )
            material_concepts[concept_entity].append(mention)

        observations = con.execute(
            "SELECT * FROM observations WHERE material_id = ? ORDER BY created_at ASC",
            (material_id_value,),
        ).fetchall()
        for observation in observations:
            observation_text = clean_text(observation["observed_text"])
            observation_evidence = source_evidence(
                material_id_value,
                "observation",
                observation["context_quote"] or observation["notes"] or observation_text,
                segment_id=observation["source_segment_id"],
                image_id=observation["source_image_id"],
                observation_id=observation["id"],
                page_ref=observation["source_page_ref"],
                source_locator=observation["source_locator"],
            )
            observation_concepts = []
            concept_entity = register_concept(observation_text, "human_observation")
            if concept_entity:
                mention = add_mention(
                    con,
                    concept_entity,
                    "concept",
                    observation_text,
                    observation_evidence,
                    "observation",
                    0.99,
                    "accepted",
                    legacy_mention_reviews,
                )
                material_concepts[concept_entity].append(mention)
                observation_concepts.append(concept_entity)
            if observation["observation_type"] == "place" and observation_text:
                place_entity = ensure_entity(
                    con,
                    "place",
                    observation_text,
                    observation_text,
                    {"resolution_source": "human_observation"},
                )
                mention = add_mention(
                    con,
                    place_entity,
                    "place",
                    observation_text,
                    observation_evidence,
                    "observation",
                    0.99,
                    "accepted",
                    legacy_mention_reviews,
                )
                upsert_place_resolution(
                    con,
                    observation_text,
                    observation_evidence,
                    "resolved",
                    0.99,
                    resolved_entity_id=place_entity,
                    notes="Accepted human place observation.",
                )
                add_relation_group(
                    relation_groups,
                    material_entity,
                    "mentions_place",
                    place_entity,
                    "spatial",
                    observation_evidence,
                    mention["confidence"],
                    mention["review_status"],
                    "human_observation",
                )
            for candidate in temporal_candidates(
                " ".join(filter(None, [observation_text, observation["context_quote"], observation["notes"]])),
                observation["source_page_ref"],
                max_items=4,
            ):
                override = time_overrides.get(candidate["normalized_label"])
                if override:
                    candidate.update(
                        {
                            "resolution_status": override["resolution_status"],
                            "start_year": override["start_year"],
                            "end_year": override["end_year"],
                        }
                    )
                time_entity = None
                if candidate["resolution_status"] == "resolved":
                    canonical = f"{candidate['time_type']}:{candidate.get('start_year')}:{candidate.get('end_year')}"
                    time_entity = ensure_entity(
                        con,
                        "time_period",
                        candidate["label"],
                        canonical,
                        {
                            "time_type": candidate["time_type"],
                            "start_year": candidate.get("start_year"),
                            "end_year": candidate.get("end_year"),
                        },
                    )
                    mention = add_mention(
                        con,
                        time_entity,
                        "time_period",
                        candidate["label"],
                        observation_evidence,
                        "observation",
                        candidate["confidence"],
                        "accepted",
                        legacy_mention_reviews,
                    )
                    add_relation_group(
                        relation_groups,
                        material_entity,
                        "mentions_time",
                        time_entity,
                        "temporal",
                        observation_evidence,
                        mention["confidence"],
                        mention["review_status"],
                        "human_observation",
                    )
                upsert_time_resolution(con, candidate, observation_evidence, time_entity)
            if len(observation_concepts) >= 2:
                unit_concepts[("observation", observation["id"])] = {
                    "concept_ids": observation_concepts,
                    "evidence": observation_evidence,
                }

        accepted_labels = set(accepted_concept_labels)
        segments = con.execute(
            """
            SELECT id, source_locator, page_ref, content_text
            FROM extracted_segments
            WHERE material_id = ?
            ORDER BY id ASC
            """,
            (material_id_value,),
        ).fetchall()
        for segment in segments:
            segment_evidence = source_evidence(
                material_id_value,
                "segment",
                segment["content_text"],
                segment_id=segment["id"],
                page_ref=segment["page_ref"],
                source_locator=segment["source_locator"],
            )
            segment_concepts = []
            for label in exact_concept_hits(segment["content_text"], accepted_labels):
                concept_entity = register_concept(label, "exact_match")
                if not concept_entity:
                    continue
                mention = add_mention(
                    con,
                    concept_entity,
                    "concept",
                    label,
                    segment_evidence,
                    "exact_match",
                    0.82,
                    "accepted",
                    legacy_mention_reviews,
                )
                material_concepts[concept_entity].append(mention)
                segment_concepts.append(concept_entity)
            if segment_concepts:
                unit_concepts[("segment", segment["id"])] = {
                    "concept_ids": segment_concepts,
                    "evidence": segment_evidence,
                }
            for place_label in extract_place_candidates(segment["content_text"], max_items=4):
                normalized = normalize_label(place_label)
                override = place_overrides.get(normalized)
                lat, lon = explicit_coordinates(place_label, segment["content_text"])
                place_entity = None
                if override and override["resolution_status"] == "resolved":
                    lat = override["latitude"]
                    lon = override["longitude"]
                    resolved_label = override["mention_label"] or place_label
                    place_entity = ensure_entity(
                        con,
                        "place",
                        resolved_label,
                        override["gazetteer_id"] or normalized,
                        {
                            "latitude": lat,
                            "longitude": lon,
                            "gazetteer_id": override["gazetteer_id"],
                            "resolution_source": "review",
                        },
                    )
                elif lat is not None and lon is not None:
                    place_entity = ensure_entity(
                        con,
                        "place",
                        place_label,
                        normalized,
                        {"latitude": lat, "longitude": lon, "resolution_source": "explicit_coordinates"},
                    )
                if place_entity:
                    mention = add_mention(
                        con,
                        place_entity,
                        "place",
                        place_label,
                        segment_evidence,
                        "pattern_candidate",
                        0.72 if lat is not None else 0.65,
                        "accepted" if override else "needs_review",
                        legacy_mention_reviews,
                    )
                    add_relation_group(
                        relation_groups,
                        material_entity,
                        "mentions_place",
                        place_entity,
                        "spatial",
                        segment_evidence,
                        mention["confidence"],
                        mention["review_status"],
                        "pattern_extraction",
                    )
                    status = "resolved"
                else:
                    status = "rejected" if override and override["resolution_status"] == "rejected" else "unresolved"
                upsert_place_resolution(
                    con,
                    place_label,
                    segment_evidence,
                    status,
                    0.5,
                    resolved_entity_id=place_entity,
                    latitude=lat,
                    longitude=lon,
                    notes="Pattern-extracted place candidate.",
                )
            for candidate in temporal_candidates(segment["content_text"], segment["page_ref"], max_items=6):
                override = time_overrides.get(candidate["normalized_label"])
                if override:
                    candidate.update(
                        {
                            "resolution_status": override["resolution_status"],
                            "start_year": override["start_year"],
                            "end_year": override["end_year"],
                        }
                    )
                time_entity = None
                if candidate["resolution_status"] == "resolved":
                    canonical = f"{candidate['time_type']}:{candidate.get('start_year')}:{candidate.get('end_year')}"
                    time_entity = ensure_entity(
                        con,
                        "time_period",
                        candidate["label"],
                        canonical,
                        {
                            "time_type": candidate["time_type"],
                            "start_year": candidate.get("start_year"),
                            "end_year": candidate.get("end_year"),
                        },
                    )
                    mention = add_mention(
                        con,
                        time_entity,
                        "time_period",
                        candidate["label"],
                        segment_evidence,
                        "pattern_candidate",
                        candidate["confidence"],
                        "needs_review",
                        legacy_mention_reviews,
                    )
                    add_relation_group(
                        relation_groups,
                        material_entity,
                        "mentions_time",
                        time_entity,
                        "temporal",
                        segment_evidence,
                        mention["confidence"],
                        mention["review_status"],
                        "pattern_extraction",
                    )
                upsert_time_resolution(con, candidate, segment_evidence, time_entity)

        images = con.execute(
            """
            SELECT id, source_locator, page_ref, ocr_text, visual_caption
            FROM image_evidence
            WHERE material_id = ?
            ORDER BY page_index ASC, created_at ASC
            """,
            (material_id_value,),
        ).fetchall()
        for image in images:
            image_text = clean_text(" ".join(filter(None, [image["ocr_text"], image["visual_caption"]])))
            image_evidence = source_evidence(
                material_id_value,
                "image",
                image_text,
                image_id=image["id"],
                page_ref=image["page_ref"],
                source_locator=image["source_locator"],
            )
            image_concepts = []
            for label in exact_concept_hits(image_text, accepted_labels):
                concept_entity = register_concept(label, "image_evidence")
                if not concept_entity:
                    continue
                method = "image_caption" if text_contains_label(image["visual_caption"], label) else "image_ocr"
                mention = add_mention(
                    con,
                    concept_entity,
                    "concept",
                    label,
                    image_evidence,
                    method,
                    0.76,
                    "accepted",
                    legacy_mention_reviews,
                )
                material_concepts[concept_entity].append(mention)
                image_concepts.append(concept_entity)
            if image_concepts:
                unit_concepts[("image", image["id"])] = {
                    "concept_ids": image_concepts,
                    "evidence": image_evidence,
                }
            for place_label in extract_place_candidates(image_text, max_items=3):
                normalized = normalize_label(place_label)
                override = place_overrides.get(normalized)
                place_entity = None
                lat = override["latitude"] if override else None
                lon = override["longitude"] if override else None
                if override and override["resolution_status"] == "resolved":
                    place_entity = ensure_entity(
                        con,
                        "place",
                        override["mention_label"] or place_label,
                        override["gazetteer_id"] or normalized,
                        {
                            "latitude": lat,
                            "longitude": lon,
                            "gazetteer_id": override["gazetteer_id"],
                            "resolution_source": "review",
                        },
                    )
                    mention = add_mention(
                        con,
                        place_entity,
                        "place",
                        place_label,
                        image_evidence,
                        "pattern_candidate",
                        0.65,
                        "accepted",
                        legacy_mention_reviews,
                    )
                    add_relation_group(
                        relation_groups,
                        material_entity,
                        "mentions_place",
                        place_entity,
                        "spatial",
                        image_evidence,
                        mention["confidence"],
                        mention["review_status"],
                        "pattern_extraction",
                    )
                upsert_place_resolution(
                    con,
                    place_label,
                    image_evidence,
                    "resolved" if place_entity else "unresolved",
                    0.45,
                    resolved_entity_id=place_entity,
                    latitude=lat,
                    longitude=lon,
                    notes="Image OCR/caption place candidate.",
                )
            for candidate in temporal_candidates(image_text, image["page_ref"], max_items=4):
                override = time_overrides.get(candidate["normalized_label"])
                if override:
                    candidate.update(
                        {
                            "resolution_status": override["resolution_status"],
                            "start_year": override["start_year"],
                            "end_year": override["end_year"],
                        }
                    )
                time_entity = None
                if candidate["resolution_status"] == "resolved":
                    canonical = f"{candidate['time_type']}:{candidate.get('start_year')}:{candidate.get('end_year')}"
                    time_entity = ensure_entity(
                        con,
                        "time_period",
                        candidate["label"],
                        canonical,
                        {
                            "time_type": candidate["time_type"],
                            "start_year": candidate.get("start_year"),
                            "end_year": candidate.get("end_year"),
                        },
                    )
                    mention = add_mention(
                        con,
                        time_entity,
                        "time_period",
                        candidate["label"],
                        image_evidence,
                        "pattern_candidate",
                        candidate["confidence"],
                        "needs_review",
                        legacy_mention_reviews,
                    )
                    add_relation_group(
                        relation_groups,
                        material_entity,
                        "mentions_time",
                        time_entity,
                        "temporal",
                        image_evidence,
                        mention["confidence"],
                        mention["review_status"],
                        "pattern_extraction",
                    )
                upsert_time_resolution(con, candidate, image_evidence, time_entity)

        for concept_entity, mentions in material_concepts.items():
            accepted_mentions = [item for item in mentions if item["review_status"] == "accepted"]
            observation_mentions = [item for item in mentions if item["mention_method"] == "observation"]
            topic_mentions = [item for item in mentions if item["mention_method"] != "observation"]
            if observation_mentions:
                for mention in observation_mentions:
                    add_relation_group(
                        relation_groups,
                        material_entity,
                        "has_observation_topic",
                        concept_entity,
                        "semantic",
                        mention,
                        mention["confidence"],
                        mention["review_status"],
                        "human_observation",
                    )
            if topic_mentions:
                for mention in topic_mentions:
                    add_relation_group(
                        relation_groups,
                        material_entity,
                        "has_topic",
                        concept_entity,
                        "semantic",
                        mention,
                        mention["confidence"],
                        "accepted" if accepted_mentions else "needs_review",
                        mention["mention_method"],
                    )

        for place_label in split_values(material["region"], max_items=4):
            normalized = normalize_label(place_label)
            override = place_overrides.get(normalized)
            lat, lon = explicit_coordinates(place_label)
            canonical = (override or {}).get("gazetteer_id") or normalized
            place_entity = ensure_entity(
                con,
                "place",
                (override or {}).get("mention_label") or place_label,
                canonical,
                {
                    "latitude": (override or {}).get("latitude") if override else lat,
                    "longitude": (override or {}).get("longitude") if override else lon,
                    "gazetteer_id": (override or {}).get("gazetteer_id"),
                    "resolution_source": "trusted_metadata",
                },
            )
            mention = add_mention(
                con,
                place_entity,
                "place",
                place_label,
                metadata_evidence,
                "metadata",
                0.95,
                "accepted",
                legacy_mention_reviews,
            )
            upsert_place_resolution(
                con,
                place_label,
                metadata_evidence,
                "resolved",
                0.95,
                resolved_entity_id=place_entity,
                latitude=(override or {}).get("latitude") if override else lat,
                longitude=(override or {}).get("longitude") if override else lon,
                gazetteer_id=(override or {}).get("gazetteer_id"),
                notes="Trusted material region metadata.",
            )
            add_relation_group(
                relation_groups,
                material_entity,
                "mentions_place",
                place_entity,
                "spatial",
                metadata_evidence,
                mention["confidence"],
                mention["review_status"],
                "metadata",
            )

        year_candidate = metadata_year_candidate(material["year"])
        if year_candidate:
            override = time_overrides.get(year_candidate["normalized_label"])
            if override:
                year_candidate.update(
                    {
                        "resolution_status": override["resolution_status"],
                        "start_year": override["start_year"],
                        "end_year": override["end_year"],
                    }
                )
            time_entity = None
            if year_candidate["resolution_status"] == "resolved":
                canonical = f"{year_candidate['time_type']}:{year_candidate.get('start_year')}:{year_candidate.get('end_year')}"
                time_entity = ensure_entity(
                    con,
                    "time_period",
                    year_candidate["label"],
                    canonical,
                    {
                        "time_type": year_candidate["time_type"],
                        "start_year": year_candidate.get("start_year"),
                        "end_year": year_candidate.get("end_year"),
                    },
                )
                mention = add_mention(
                    con,
                    time_entity,
                    "time_period",
                    year_candidate["label"],
                    metadata_evidence,
                    "metadata",
                    0.98,
                    "accepted",
                    legacy_mention_reviews,
                )
                add_relation_group(
                    relation_groups,
                    material_entity,
                    "mentions_time",
                    time_entity,
                    "temporal",
                    metadata_evidence,
                    mention["confidence"],
                    mention["review_status"],
                    "metadata",
                )
            upsert_time_resolution(con, year_candidate, metadata_evidence, time_entity)

    persist_relation_groups(con, relation_groups, review_decisions)
    preliminary_stats = compute_entity_stats(con)
    persist_cooccurrence_candidates(
        con,
        unit_concepts,
        preliminary_stats,
        review_decisions,
        legacy_candidate_reviews,
    )
    promote_accepted_candidates(con)
    compute_entity_stats(con)

    counts = {
        "material_count": len(materials),
        "entity_count": con.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0],
        "relation_count": con.execute("SELECT COUNT(*) FROM kg_relations").fetchone()[0],
        "mention_count": con.execute("SELECT COUNT(*) FROM kg_mentions").fetchone()[0],
        "candidate_count": con.execute("SELECT COUNT(*) FROM kg_candidate_relations").fetchone()[0],
        "unresolved_place_count": con.execute(
            "SELECT COALESCE(SUM(evidence_count), 0) FROM kg_place_resolution WHERE resolution_status IN ('unresolved', 'ambiguous')"
        ).fetchone()[0],
        "review_time_count": con.execute(
            "SELECT COALESCE(SUM(evidence_count), 0) FROM kg_time_resolution WHERE resolution_status IN ('invalid', 'needs_review', 'ambiguous')"
        ).fetchone()[0],
    }
    return counts


def semantic_entity_from_row(row: sqlite3.Row) -> dict:
    properties = parse_json(row["properties_json"], {})
    return {
        "id": row["id"],
        "type": row["entity_type"],
        "label": row["label"],
        "normalized_label": row["normalized_label"],
        "canonical_key": row["canonical_key"],
        "importance_score": float(row["importance_score"] or 0),
        "mention_count": int(row["mention_count"] or 0),
        "document_count": int(row["document_count"] or 0),
        "collection_count": int(row["collection_count"] or 0),
        "degree": int(row["degree"] or 0),
        "accepted_relation_count": int(row["accepted_relation_count"] or 0),
        "review_relation_count": int(row["review_relation_count"] or 0),
        "evidence_count": int(row["mention_count"] or 0),
        "is_generic": bool(row["is_generic"]),
        "is_bridge_entity": bool(row["is_bridge_entity"]),
        "properties": properties,
    }


def relation_evidence_preview(con: sqlite3.Connection, item_id: str) -> dict:
    row = con.execute(
        """
        SELECT re.*, m.title AS material_title
        FROM kg_relation_evidence re
        JOIN materials m ON m.id = re.material_id
        WHERE re.relation_id = ?
        ORDER BY re.created_at ASC
        LIMIT 1
        """,
        (item_id,),
    ).fetchone()
    if not row:
        return {}
    return {
        "material_id": row["material_id"],
        "material_title": row["material_title"],
        "segment_id": row["segment_id"],
        "image_id": row["image_id"],
        "observation_id": row["observation_id"],
        "page_ref": row["page_ref"],
        "source_locator": row["source_locator"],
        "source": row["evidence_type"],
        "snippet": row["snippet"],
    }


def semantic_relation_from_row(con: sqlite3.Connection, row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "source": row["subject_entity_id"],
        "target": row["object_entity_id"],
        "predicate": row["predicate"],
        "relation_family": row["relation_family"],
        "evidence_count": int(row["evidence_count"]),
        "document_count": int(row["document_count"]),
        "confidence": float(row["confidence"]),
        "status": row["assertion_status"],
        "extraction_method": row["extraction_method"],
        "properties": parse_json(row["properties_json"], {}),
        "evidence_preview": (
            f"{row['evidence_count']} evidence item"
            f"{'' if row['evidence_count'] == 1 else 's'} support this relationship."
        ),
        "evidence_ref": relation_evidence_preview(con, row["id"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def semantic_graph_payload(
    con: sqlite3.Connection,
    query: Optional[str] = None,
    view: str = "overview",
    include_generic: bool = False,
    include_rejected: bool = False,
    limit: int = 180,
) -> dict:
    init_semantic_graph_schema(con)
    normalized_query = normalize_label(query)
    base_sql = """
        SELECT e.*, COALESCE(s.mention_count, 0) AS mention_count,
               COALESCE(s.document_count, 0) AS document_count,
               COALESCE(s.collection_count, 0) AS collection_count,
               COALESCE(s.degree, 0) AS degree,
               COALESCE(s.accepted_relation_count, 0) AS accepted_relation_count,
               COALESCE(s.review_relation_count, 0) AS review_relation_count,
               COALESCE(s.importance_score, 0) AS importance_score
        FROM kg_entities e
        LEFT JOIN kg_entity_stats s ON s.entity_id = e.id
    """
    selected_ids: set[str] = set()
    if normalized_query:
        rows = con.execute(
            base_sql
            + """
            WHERE e.normalized_label LIKE ?
               OR lower(e.properties_json) LIKE lower(?)
            ORDER BY importance_score DESC, document_count DESC
            LIMIT ?
            """,
            (f"%{normalized_query}%", f"%{query or ''}%", min(limit, 60)),
        ).fetchall()
        selected_ids.update(row["id"] for row in rows)
        if selected_ids:
            placeholders = ",".join("?" for _ in selected_ids)
            neighbor_rows = con.execute(
                f"""
                SELECT subject_entity_id, object_entity_id
                FROM kg_relations
                WHERE assertion_status != 'rejected'
                  AND (subject_entity_id IN ({placeholders}) OR object_entity_id IN ({placeholders}))
                ORDER BY confidence DESC, evidence_count DESC
                LIMIT ?
                """,
                (*selected_ids, *selected_ids, limit * 2),
            ).fetchall()
            for row in neighbor_rows:
                selected_ids.add(row["subject_entity_id"])
                selected_ids.add(row["object_entity_id"])
    else:
        view_types = {
            "overview": {"collection", "source_type", "material", "concept", "place", "time_period", "agent"},
            "documents": {"collection", "source_type", "material", "agent"},
            "concepts": {"material", "concept"},
            "places_time": {"material", "place", "time_period"},
        }.get(view, {"collection", "source_type", "material", "concept", "place", "time_period", "agent"})
        rows = con.execute(
            base_sql
            + """
            ORDER BY
                CASE e.entity_type
                    WHEN 'collection' THEN 0
                    WHEN 'source_type' THEN 1
                    WHEN 'material' THEN 2
                    WHEN 'concept' THEN 3
                    WHEN 'place' THEN 4
                    WHEN 'time_period' THEN 5
                    ELSE 6
                END,
                importance_score DESC,
                document_count DESC,
                lower(e.label)
            """
        ).fetchall()
        type_caps = {
            "collection": 24,
            "source_type": 16,
            "material": 40,
            "concept": 36,
            "place": 18,
            "time_period": 18,
            "agent": 16,
        }
        if view == "overview":
            type_caps.update(
                {
                    "collection": 12,
                    "source_type": 8,
                    "material": 24,
                    "concept": 10,
                    "place": 6,
                    "time_period": 6,
                    "agent": 6,
                }
            )
        elif view == "documents":
            type_caps.update({"collection": 16, "source_type": 12, "material": 50, "agent": 20})
        elif view == "places_time":
            type_caps.update({"material": 40, "place": 30, "time_period": 30})
        type_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            entity_type = row["entity_type"]
            if entity_type not in view_types:
                continue
            if row["is_generic"] and not include_generic:
                continue
            if entity_type == "concept" and not row["is_bridge_entity"]:
                continue
            if entity_type == "agent" and int(row["document_count"] or 0) < 2:
                continue
            if type_counts[entity_type] >= type_caps[entity_type]:
                continue
            selected_ids.add(row["id"])
            type_counts[entity_type] += 1
            if len(selected_ids) >= limit:
                break

    if not selected_ids:
        return {
            "query": query,
            "view": view,
            "nodes": [],
            "edges": [],
            "summary": {f"{entity_type}_count": 0 for entity_type in ENTITY_TYPES} | {"relation_count": 0},
            "hidden_summary": semantic_hidden_summary(con),
            "evidence_note": "This view shows the simplified semantic graph. Evidence details are available on click.",
        }

    placeholders = ",".join("?" for _ in selected_ids)
    node_rows = con.execute(
        base_sql
        + f"""
        WHERE e.id IN ({placeholders})
        ORDER BY importance_score DESC, document_count DESC, lower(e.label)
        """,
        tuple(selected_ids),
    ).fetchall()
    status_clause = "" if include_rejected else "AND assertion_status != 'rejected'"
    relation_rows = con.execute(
        f"""
        SELECT *
        FROM kg_relations
        WHERE subject_entity_id IN ({placeholders})
          AND object_entity_id IN ({placeholders})
          {status_clause}
        ORDER BY
            CASE assertion_status WHEN 'accepted' THEN 0 WHEN 'needs_review' THEN 1 ELSE 2 END,
            confidence DESC, evidence_count DESC
        LIMIT ?
        """,
        (*selected_ids, *selected_ids, limit * 5),
    ).fetchall()
    connected_ids = {
        node_id
        for row in relation_rows
        for node_id in (row["subject_entity_id"], row["object_entity_id"])
    }
    nodes = [
        semantic_entity_from_row(row)
        for row in node_rows
        if row["entity_type"] == "material" or row["id"] in connected_ids
    ]
    edges = [semantic_relation_from_row(con, row) for row in relation_rows]
    summary = {f"{entity_type}_count": 0 for entity_type in ENTITY_TYPES}
    for node in nodes:
        summary[f"{node['type']}_count"] += 1
    summary["relation_count"] = len(edges)
    return {
        "query": query,
        "view": view,
        "nodes": nodes,
        "edges": edges,
        "summary": summary,
        "hidden_summary": semantic_hidden_summary(con),
        "evidence_note": "This view shows the simplified semantic graph. Evidence details are available on click.",
    }


def semantic_hidden_summary(con: sqlite3.Connection) -> dict:
    return {
        "hidden_mentions": con.execute("SELECT COUNT(*) FROM kg_mentions").fetchone()[0],
        "hidden_candidate_relations": con.execute(
            "SELECT COUNT(*) FROM kg_candidate_relations WHERE review_status != 'rejected'"
        ).fetchone()[0],
        "hidden_cooccurrence_candidates": con.execute(
            "SELECT COUNT(*) FROM kg_candidate_relations WHERE candidate_method = 'cooccurrence' AND review_status != 'rejected'"
        ).fetchone()[0],
        "unresolved_place_mentions": con.execute(
            "SELECT COALESCE(SUM(evidence_count), 0) FROM kg_place_resolution WHERE resolution_status IN ('unresolved', 'ambiguous')"
        ).fetchone()[0],
        "invalid_or_candidate_time_mentions": con.execute(
            "SELECT COALESCE(SUM(evidence_count), 0) FROM kg_time_resolution WHERE resolution_status IN ('invalid', 'needs_review', 'ambiguous')"
        ).fetchone()[0],
        "generic_entities_hidden": con.execute(
            "SELECT COUNT(*) FROM kg_entities WHERE is_generic = 1"
        ).fetchone()[0],
    }


def get_entity_row(con: sqlite3.Connection, item_id: str) -> sqlite3.Row:
    row = con.execute(
        """
        SELECT e.*, COALESCE(s.mention_count, 0) AS mention_count,
               COALESCE(s.document_count, 0) AS document_count,
               COALESCE(s.collection_count, 0) AS collection_count,
               COALESCE(s.degree, 0) AS degree,
               COALESCE(s.accepted_relation_count, 0) AS accepted_relation_count,
               COALESCE(s.review_relation_count, 0) AS review_relation_count,
               COALESCE(s.importance_score, 0) AS importance_score
        FROM kg_entities e
        LEFT JOIN kg_entity_stats s ON s.entity_id = e.id
        WHERE e.id = ?
        """,
        (item_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Semantic graph entity not found")
    return row


def semantic_entity_detail(con: sqlite3.Connection, item_id: str, include_rejected: bool = False) -> dict:
    entity = semantic_entity_from_row(get_entity_row(con, item_id))
    status_clause = "" if include_rejected else "AND r.assertion_status != 'rejected'"
    rows = con.execute(
        f"""
        SELECT r.*
        FROM kg_relations r
        WHERE (r.subject_entity_id = ? OR r.object_entity_id = ?)
          {status_clause}
        ORDER BY r.confidence DESC, r.evidence_count DESC
        LIMIT 100
        """,
        (item_id, item_id),
    ).fetchall()
    relations = [semantic_relation_from_row(con, row) for row in rows]
    neighbor_ids = {
        row["object_entity_id"] if row["subject_entity_id"] == item_id else row["subject_entity_id"]
        for row in rows
    }
    connected = []
    if neighbor_ids:
        placeholders = ",".join("?" for _ in neighbor_ids)
        neighbor_rows = con.execute(
            f"""
            SELECT e.*, COALESCE(s.mention_count, 0) AS mention_count,
                   COALESCE(s.document_count, 0) AS document_count,
                   COALESCE(s.collection_count, 0) AS collection_count,
                   COALESCE(s.degree, 0) AS degree,
                   COALESCE(s.accepted_relation_count, 0) AS accepted_relation_count,
                   COALESCE(s.review_relation_count, 0) AS review_relation_count,
                   COALESCE(s.importance_score, 0) AS importance_score
            FROM kg_entities e
            LEFT JOIN kg_entity_stats s ON s.entity_id = e.id
            WHERE e.id IN ({placeholders})
            ORDER BY importance_score DESC, lower(e.label)
            """,
            tuple(neighbor_ids),
        ).fetchall()
        connected = [semantic_entity_from_row(row) for row in neighbor_rows]
    candidates = semantic_candidates(con, entity_id_value=item_id, limit=20)["candidates"]
    return {
        "entity": entity,
        "relations": relations,
        "connected_entities": connected,
        "related_materials": semantic_related_materials(con, item_id, limit=20)["materials"],
        "candidates": candidates,
        "evidence_note": "Semantic entities summarize accepted and reviewable assertions. Source evidence remains available separately.",
    }


def semantic_entity_evidence(con: sqlite3.Connection, item_id: str, limit: int = 50, offset: int = 0) -> dict:
    entity = semantic_entity_from_row(get_entity_row(con, item_id))
    material_id_value = entity["properties"].get("material_id") if entity["type"] == "material" else None
    where = "km.material_id = ?" if material_id_value else "km.entity_id = ?"
    value = material_id_value or item_id
    total = con.execute(f"SELECT COUNT(*) FROM kg_mentions km WHERE {where}", (value,)).fetchone()[0]
    rows = con.execute(
        f"""
        SELECT km.*, m.title AS material_title
        FROM kg_mentions km
        JOIN materials m ON m.id = km.material_id
        WHERE {where}
        ORDER BY
            CASE km.review_status WHEN 'accepted' THEN 0 WHEN 'needs_review' THEN 1 ELSE 2 END,
            km.confidence DESC, km.created_at DESC
        LIMIT ? OFFSET ?
        """,
        (value, limit, offset),
    ).fetchall()
    return {
        "entity": entity,
        "items": [
            {
                **dict(row),
                "evidence_ref": {
                    "material_id": row["material_id"],
                    "material_title": row["material_title"],
                    "segment_id": row["segment_id"],
                    "image_id": row["image_id"],
                    "observation_id": row["observation_id"],
                    "page_ref": row["page_ref"],
                    "source_locator": row["source_locator"],
                    "source": row["evidence_type"],
                    "snippet": row["snippet"],
                },
            }
            for row in rows
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def semantic_related_materials(con: sqlite3.Connection, item_id: str, limit: int = 30) -> dict:
    entity = semantic_entity_from_row(get_entity_row(con, item_id))
    materials = []
    if entity["type"] == "material":
        concept_rows = con.execute(
            """
            SELECT object_entity_id
            FROM kg_relations
            WHERE subject_entity_id = ?
              AND predicate IN ('has_topic', 'has_observation_topic')
              AND assertion_status != 'rejected'
            """,
            (item_id,),
        ).fetchall()
        concept_ids = [row["object_entity_id"] for row in concept_rows]
        if concept_ids:
            placeholders = ",".join("?" for _ in concept_ids)
            rows = con.execute(
                f"""
                SELECT e.id, e.label, e.properties_json,
                       COUNT(DISTINCT r.object_entity_id) AS shared_concept_count,
                       SUM(r.evidence_count) AS evidence_count
                FROM kg_relations r
                JOIN kg_entities e ON e.id = r.subject_entity_id
                WHERE r.object_entity_id IN ({placeholders})
                  AND r.subject_entity_id != ?
                  AND r.predicate IN ('has_topic', 'has_observation_topic')
                  AND r.assertion_status != 'rejected'
                GROUP BY e.id
                ORDER BY shared_concept_count DESC, evidence_count DESC
                LIMIT ?
                """,
                (*concept_ids, item_id, limit),
            ).fetchall()
            materials = [
                {
                    "entity_id": row["id"],
                    "material_id": parse_json(row["properties_json"], {}).get("material_id"),
                    "title": row["label"],
                    "shared_concept_count": row["shared_concept_count"],
                    "evidence_count": row["evidence_count"],
                }
                for row in rows
            ]
    else:
        rows = con.execute(
            """
            SELECT e.id, e.label, e.properties_json, r.evidence_count, r.confidence
            FROM kg_relations r
            JOIN kg_entities e ON e.id = r.subject_entity_id
            WHERE r.object_entity_id = ?
              AND e.entity_type = 'material'
              AND r.assertion_status != 'rejected'
            ORDER BY r.evidence_count DESC, r.confidence DESC, lower(e.label)
            LIMIT ?
            """,
            (item_id, limit),
        ).fetchall()
        materials = [
            {
                "entity_id": row["id"],
                "material_id": parse_json(row["properties_json"], {}).get("material_id"),
                "title": row["label"],
                "shared_concept_count": 1,
                "evidence_count": row["evidence_count"],
            }
            for row in rows
        ]
    return {"entity": entity, "materials": materials, "count": len(materials)}


def semantic_relation_evidence(con: sqlite3.Connection, item_id: str, limit: int = 50, offset: int = 0) -> dict:
    row = con.execute("SELECT * FROM kg_relations WHERE id = ?", (item_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Semantic graph relation not found")
    total = con.execute(
        "SELECT COUNT(*) FROM kg_relation_evidence WHERE relation_id = ?",
        (item_id,),
    ).fetchone()[0]
    rows = con.execute(
        """
        SELECT re.*, m.title AS material_title
        FROM kg_relation_evidence re
        JOIN materials m ON m.id = re.material_id
        WHERE re.relation_id = ?
        ORDER BY re.created_at ASC
        LIMIT ? OFFSET ?
        """,
        (item_id, limit, offset),
    ).fetchall()
    return {
        "relation": semantic_relation_from_row(con, row),
        "items": [
            {
                "id": item["id"],
                "evidence_type": item["evidence_type"],
                "evidence_ref": {
                    "material_id": item["material_id"],
                    "material_title": item["material_title"],
                    "segment_id": item["segment_id"],
                    "image_id": item["image_id"],
                    "observation_id": item["observation_id"],
                    "page_ref": item["page_ref"],
                    "source_locator": item["source_locator"],
                    "source": item["evidence_type"],
                    "snippet": item["snippet"],
                },
            }
            for item in rows
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def candidate_from_row(con: sqlite3.Connection, row: sqlite3.Row) -> dict:
    labels = con.execute(
        "SELECT id, label, entity_type FROM kg_entities WHERE id IN (?, ?)",
        (row["subject_entity_id"], row["object_entity_id"]),
    ).fetchall()
    label_map = {item["id"]: dict(item) for item in labels}
    return {
        "id": row["id"],
        "source": row["subject_entity_id"],
        "target": row["object_entity_id"],
        "source_entity": label_map.get(row["subject_entity_id"]),
        "target_entity": label_map.get(row["object_entity_id"]),
        "predicate": row["predicate"],
        "candidate_method": row["candidate_method"],
        "evidence_count": row["evidence_count"],
        "document_count": row["document_count"],
        "confidence": row["confidence"],
        "review_status": row["review_status"],
        "rationale": row["rationale"],
        "properties": parse_json(row["properties_json"], {}),
    }


def semantic_candidates(
    con: sqlite3.Connection,
    status: Optional[str] = None,
    entity_id_value: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 50,
) -> dict:
    where = []
    params: list[object] = []
    if status:
        where.append("c.review_status = ?")
        params.append(status)
    if entity_id_value:
        where.append("(c.subject_entity_id = ? OR c.object_entity_id = ?)")
        params.extend([entity_id_value, entity_id_value])
    if query:
        where.append("(lower(source.label) LIKE lower(?) OR lower(target.label) LIKE lower(?) OR lower(c.rationale) LIKE lower(?))")
        like = f"%{query}%"
        params.extend([like, like, like])
    rows = con.execute(
        f"""
        SELECT c.*
        FROM kg_candidate_relations c
        JOIN kg_entities source ON source.id = c.subject_entity_id
        JOIN kg_entities target ON target.id = c.object_entity_id
        {'WHERE ' + ' AND '.join(where) if where else ''}
        ORDER BY
            CASE c.review_status WHEN 'needs_review' THEN 0 WHEN 'accepted' THEN 1 ELSE 2 END,
            c.confidence DESC, c.evidence_count DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return {
        "candidates": [candidate_from_row(con, row) for row in rows],
        "count": len(rows),
        "evidence_note": "Candidate relationships are analytical suggestions and do not appear in the Semantic Atlas until accepted.",
    }


def resolution_review_payload(
    con: sqlite3.Connection,
    kind: str,
    status: Optional[str] = None,
    limit: int = 100,
) -> dict:
    table_name = "kg_place_resolution" if kind == "place" else "kg_time_resolution"
    where = "WHERE resolution_status = ?" if status else ""
    params: tuple = (status, limit) if status else (limit,)
    rows = con.execute(
        f"""
        SELECT *
        FROM {table_name}
        {where}
        ORDER BY evidence_count DESC, lower(mention_label)
        LIMIT ?
        """,
        params,
    ).fetchall()
    items = []
    for row in rows:
        evidence_rows = con.execute(
            """
            SELECT re.*, m.title AS material_title
            FROM kg_resolution_evidence re
            JOIN materials m ON m.id = re.material_id
            WHERE re.resolution_kind = ? AND re.resolution_id = ?
            ORDER BY re.created_at ASC
            LIMIT 3
            """,
            (kind, row["id"]),
        ).fetchall()
        items.append(
            {
                **dict(row),
                "evidence": [
                    {
                        "material_id": evidence["material_id"],
                        "material_title": evidence["material_title"],
                        "segment_id": evidence["segment_id"],
                        "image_id": evidence["image_id"],
                        "observation_id": evidence["observation_id"],
                        "page_ref": evidence["page_ref"],
                        "source_locator": evidence["source_locator"],
                        "source": evidence["evidence_type"],
                        "snippet": evidence["snippet"],
                    }
                    for evidence in evidence_rows
                ],
            }
        )
    return {"items": items, "count": len(items), "kind": kind}


def save_review_decision(
    con: sqlite3.Connection,
    target_kind: str,
    target_key: str,
    decision: str,
    notes: Optional[str] = None,
):
    con.execute(
        """
        INSERT INTO kg_review_decisions (id, target_kind, target_key, decision, notes, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(target_kind, target_key) DO UPDATE SET
            decision = excluded.decision,
            notes = excluded.notes,
            updated_at = excluded.updated_at
        """,
        (
            stable_id("kgd", target_kind, target_key),
            target_kind,
            target_key,
            decision,
            notes,
            now_iso(),
        ),
    )


def review_semantic_relation(con: sqlite3.Connection, item_id: str, status: str) -> dict:
    if status not in REVIEW_STATUSES:
        raise HTTPException(status_code=422, detail="Invalid semantic relation review status")
    row = con.execute("SELECT * FROM kg_relations WHERE id = ?", (item_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Semantic graph relation not found")
    con.execute(
        "UPDATE kg_relations SET assertion_status = ?, updated_at = ? WHERE id = ?",
        (status, now_iso(), item_id),
    )
    save_review_decision(con, "relation", item_id, status)
    updated = con.execute("SELECT * FROM kg_relations WHERE id = ?", (item_id,)).fetchone()
    return semantic_relation_from_row(con, updated)


def review_semantic_candidate(con: sqlite3.Connection, item_id: str, status: str) -> dict:
    if status not in REVIEW_STATUSES:
        raise HTTPException(status_code=422, detail="Invalid semantic candidate review status")
    row = con.execute("SELECT * FROM kg_candidate_relations WHERE id = ?", (item_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Semantic candidate relation not found")
    con.execute(
        "UPDATE kg_candidate_relations SET review_status = ?, updated_at = ? WHERE id = ?",
        (status, now_iso(), item_id),
    )
    save_review_decision(con, "candidate", item_id, status)
    if status == "accepted":
        promote_accepted_candidates(con)
        promoted_id = relation_id(row["subject_entity_id"], "related_to", row["object_entity_id"])
        save_review_decision(con, "relation", promoted_id, "accepted")
    compute_entity_stats(con)
    updated = con.execute("SELECT * FROM kg_candidate_relations WHERE id = ?", (item_id,)).fetchone()
    return candidate_from_row(con, updated)


def review_place_resolution(
    con: sqlite3.Connection,
    item_id: str,
    status: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    resolved_label: Optional[str] = None,
    gazetteer_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict:
    if status not in {"resolved", "ambiguous", "unresolved", "rejected"}:
        raise HTTPException(status_code=422, detail="Invalid place resolution status")
    row = con.execute("SELECT * FROM kg_place_resolution WHERE id = ?", (item_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Place resolution not found")
    place_entity = row["resolved_place_entity_id"]
    if status == "resolved":
        label = clean_text(resolved_label or row["mention_label"])
        place_entity = ensure_entity(
            con,
            "place",
            label,
            gazetteer_id or normalize_label(label),
            {
                "latitude": latitude,
                "longitude": longitude,
                "gazetteer_id": gazetteer_id,
                "resolution_source": "review",
            },
        )
    con.execute(
        """
        UPDATE kg_place_resolution
        SET mention_label = ?, resolved_place_entity_id = ?, latitude = ?,
            longitude = ?, gazetteer_id = ?, resolution_status = ?, notes = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            resolved_label or row["mention_label"],
            place_entity if status == "resolved" else None,
            latitude,
            longitude,
            gazetteer_id,
            status,
            notes,
            now_iso(),
            item_id,
        ),
    )
    save_review_decision(con, "place_resolution", row["normalized_label"], status, notes)
    return dict(con.execute("SELECT * FROM kg_place_resolution WHERE id = ?", (item_id,)).fetchone())


def review_time_resolution(
    con: sqlite3.Connection,
    item_id: str,
    status: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    resolved_label: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict:
    if status not in {"resolved", "ambiguous", "invalid", "needs_review", "rejected"}:
        raise HTTPException(status_code=422, detail="Invalid time resolution status")
    row = con.execute("SELECT * FROM kg_time_resolution WHERE id = ?", (item_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Time resolution not found")
    time_entity = row["resolved_time_entity_id"]
    if status == "resolved":
        if start_year is None:
            raise HTTPException(status_code=422, detail="Resolved time requires a start year")
        final_end = end_year if end_year is not None else start_year
        label = clean_text(resolved_label or row["mention_label"])
        time_entity = ensure_entity(
            con,
            "time_period",
            label,
            f"historical_period:{start_year}:{final_end}",
            {"time_type": "historical_period", "start_year": start_year, "end_year": final_end},
        )
        end_year = final_end
    con.execute(
        """
        UPDATE kg_time_resolution
        SET mention_label = ?, resolved_time_entity_id = ?, start_year = ?,
            end_year = ?, resolution_status = ?, notes = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            resolved_label or row["mention_label"],
            time_entity if status == "resolved" else None,
            start_year,
            end_year,
            status,
            notes,
            now_iso(),
            item_id,
        ),
    )
    save_review_decision(con, "time_resolution", row["normalized_label"], status, notes)
    return dict(con.execute("SELECT * FROM kg_time_resolution WHERE id = ?", (item_id,)).fetchone())


def legacy_edge_from_semantic_relation(con: sqlite3.Connection, row: sqlite3.Row) -> dict:
    evidence_ref = relation_evidence_preview(con, row["id"])
    return {
        "id": row["id"],
        "source_node_id": row["subject_entity_id"],
        "target_node_id": row["object_entity_id"],
        "edge_type": row["predicate"],
        "weight": max(0.5, min(4.0, math.log1p(row["evidence_count"]))),
        "confidence": row["confidence"],
        "evidence_ref": evidence_ref,
        "extraction_method": row["extraction_method"],
        "review_status": row["assertion_status"],
        "created_at": row["created_at"],
    }


def semantic_timeline(
    con: sqlite3.Connection,
    query: Optional[str] = None,
    material_id: Optional[str] = None,
    limit: int = 100,
) -> dict:
    params: list[object] = []
    where = [
        "r.predicate = 'mentions_time'",
        "r.assertion_status != 'rejected'",
        "subject.entity_type = 'material'",
        "target.entity_type = 'time_period'",
    ]
    if material_id:
        where.append("json_extract(subject.properties_json, '$.material_id') = ?")
        params.append(material_id)
    if query:
        where.append("(lower(subject.label) LIKE lower(?) OR lower(target.label) LIKE lower(?))")
        params.extend([f"%{query}%", f"%{query}%"])
    valid_total = con.execute(
        f"""
        SELECT COUNT(*)
        FROM kg_relations r
        JOIN kg_entities subject ON subject.id = r.subject_entity_id
        JOIN kg_entities target ON target.id = r.object_entity_id
        WHERE {' AND '.join(where)}
        """,
        tuple(params),
    ).fetchone()[0]
    rows = con.execute(
        f"""
        SELECT r.*, subject.label AS source_label, target.label AS time_label,
               target.properties_json AS time_properties
        FROM kg_relations r
        JOIN kg_entities subject ON subject.id = r.subject_entity_id
        JOIN kg_entities target ON target.id = r.object_entity_id
        WHERE {' AND '.join(where)}
        ORDER BY CAST(json_extract(target.properties_json, '$.start_year') AS INTEGER), target.label
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    items = []
    for row in rows:
        props = parse_json(row["time_properties"], {})
        items.append(
            {
                "time_label": row["time_label"],
                "sort_year": props.get("start_year"),
                "source_node_id": row["subject_entity_id"],
                "source_label": row["source_label"],
                "source_type": "material",
                "edge": legacy_edge_from_semantic_relation(con, row),
            }
        )
    unresolved_rows = con.execute(
        """
        SELECT tr.*, re.material_id, re.evidence_type, re.segment_id, re.image_id,
               re.observation_id, re.page_ref, re.source_locator, re.snippet,
               m.title AS material_title
        FROM kg_time_resolution tr
        JOIN kg_resolution_evidence re
          ON re.resolution_kind = 'time' AND re.resolution_id = tr.id
        JOIN materials m ON m.id = re.material_id
        WHERE tr.resolution_status IN ('invalid', 'needs_review', 'ambiguous')
        ORDER BY tr.evidence_count DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    unresolved = []
    for row in unresolved_rows:
        if material_id and row["material_id"] != material_id:
            continue
        if query and normalize_label(query) not in normalize_label(
            f"{row['mention_label']} {row['material_title']} {row['snippet'] or ''}"
        ):
            continue
        unresolved.append(
            {
                "time_label": row["mention_label"],
                "sort_year": None,
                "source_node_id": row["material_id"],
                "source_label": row["material_title"],
                "source_type": "material",
                "edge": {
                    "id": row["id"],
                    "source_node_id": row["material_id"],
                    "target_node_id": row["id"],
                    "edge_type": "pattern_mentions_time",
                    "weight": 0.4,
                    "confidence": row["confidence"] or 0,
                    "evidence_ref": {
                        "material_id": row["material_id"],
                        "material_title": row["material_title"],
                        "segment_id": row["segment_id"],
                        "image_id": row["image_id"],
                        "observation_id": row["observation_id"],
                        "page_ref": row["page_ref"],
                        "source_locator": row["source_locator"],
                        "source": row["evidence_type"],
                        "snippet": row["snippet"],
                    },
                    "extraction_method": "pattern_extraction",
                    "review_status": "needs_review",
                    "created_at": row["updated_at"],
                },
            }
        )
    review_where = ["tr.resolution_status IN ('invalid', 'needs_review', 'ambiguous')"]
    review_params: list[object] = []
    if material_id:
        review_where.append("re.material_id = ?")
        review_params.append(material_id)
    if query:
        review_where.append(
            "(lower(tr.mention_label) LIKE lower(?) OR lower(m.title) LIKE lower(?) OR lower(re.snippet) LIKE lower(?))"
        )
        review_params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])
    review_total = con.execute(
        f"""
        SELECT COUNT(*)
        FROM kg_time_resolution tr
        JOIN kg_resolution_evidence re
          ON re.resolution_kind = 'time' AND re.resolution_id = tr.id
        JOIN materials m ON m.id = re.material_id
        WHERE {' AND '.join(review_where)}
        """,
        tuple(review_params),
    ).fetchone()[0]
    return {
        "query": query,
        "material_id": material_id,
        "items": items,
        "unresolved": unresolved,
        "summary": {
            "valid_time_count": valid_total,
            "review_time_count": review_total,
        },
        "evidence_note": "Time Evidence shows validated periods separately from invalid or ambiguous date candidates.",
    }


def semantic_map(
    con: sqlite3.Connection,
    query: Optional[str] = None,
    material_id: Optional[str] = None,
    limit: int = 100,
) -> dict:
    params: list[object] = []
    where = [
        "r.predicate = 'mentions_place'",
        "r.assertion_status != 'rejected'",
        "subject.entity_type = 'material'",
        "target.entity_type = 'place'",
    ]
    if material_id:
        where.append("json_extract(subject.properties_json, '$.material_id') = ?")
        params.append(material_id)
    if query:
        where.append("(lower(subject.label) LIKE lower(?) OR lower(target.label) LIKE lower(?))")
        params.extend([f"%{query}%", f"%{query}%"])
    rows = con.execute(
        f"""
        SELECT r.*, subject.label AS source_label, target.label AS place_label,
               target.properties_json AS place_properties
        FROM kg_relations r
        JOIN kg_entities subject ON subject.id = r.subject_entity_id
        JOIN kg_entities target ON target.id = r.object_entity_id
        WHERE {' AND '.join(where)}
        ORDER BY r.confidence DESC, r.evidence_count DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    features = []
    accepted_without_coordinates = []
    for row in rows:
        properties = parse_json(row["place_properties"], {})
        edge = legacy_edge_from_semantic_relation(con, row)
        item = {
            "place_label": row["place_label"],
            "source_node_id": row["subject_entity_id"],
            "source_label": row["source_label"],
            "source_type": "material",
            "edge": edge,
        }
        lat = properties.get("latitude")
        lon = properties.get("longitude")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "place_label": row["place_label"],
                        "source_label": row["source_label"],
                        "source_type": "material",
                        "edge_id": row["id"],
                        "review_status": row["assertion_status"],
                        "confidence": row["confidence"],
                        "evidence_ref": edge["evidence_ref"],
                    },
                }
            )
        else:
            accepted_without_coordinates.append(item)
    review_payload = resolution_review_payload(con, "place", limit=limit)
    unresolved = list(accepted_without_coordinates)
    for item in review_payload["items"]:
        if item["resolution_status"] not in {"unresolved", "ambiguous"}:
            continue
        evidence = item["evidence"][0] if item["evidence"] else {}
        if material_id and evidence.get("material_id") != material_id:
            continue
        if query and normalize_label(query) not in normalize_label(
            f"{item['mention_label']} {evidence.get('material_title', '')} {evidence.get('snippet', '')}"
        ):
            continue
        unresolved.append(
            {
                "place_label": item["mention_label"],
                "source_node_id": evidence.get("material_id", item["id"]),
                "source_label": evidence.get("material_title", "Unresolved place"),
                "source_type": evidence.get("source", "pattern_candidate"),
                "edge": {
                    "id": item["id"],
                    "source_node_id": evidence.get("material_id", item["id"]),
                    "target_node_id": item["id"],
                    "edge_type": "pattern_mentions_place",
                    "weight": 0.4,
                    "confidence": item["confidence"] or 0,
                    "evidence_ref": evidence,
                    "extraction_method": "pattern_extraction",
                    "review_status": "needs_review",
                    "created_at": item["updated_at"],
                },
            }
        )
    return {
        "query": query,
        "material_id": material_id,
        "geojson": {"type": "FeatureCollection", "features": features},
        "unresolved": unresolved[:limit],
        "summary": {
            "resolved_coordinate_count": len(features),
            "unresolved_place_mentions": con.execute(
                """
                SELECT COALESCE(SUM(evidence_count), 0)
                FROM kg_place_resolution
                WHERE resolution_status IN ('unresolved', 'ambiguous')
                """
            ).fetchone()[0],
            "accepted_without_coordinates": len(accepted_without_coordinates),
        },
        "evidence_note": "Place Evidence maps coordinate-resolved places and keeps unresolved labels in a separate review list.",
    }
