"""Versioned, full-coverage embeddings in the existing JSON vector columns.

Offsets refer to the unchanged parent string. Legacy list vectors remain readable;
only a complete span document with the current content hash counts as current.
"""

import hashlib
import json
import math
import re

FORMAT = "repository-span-embeddings-v2"
SPAN_CHARS = 2000
SPAN_OVERLAP = 160
INDEX_METHOD = f"spans-v2:{SPAN_CHARS}:{SPAN_OVERLAP}:RETRIEVAL_DOCUMENT"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_spans(text: str) -> list[dict]:
    spans = []
    start = 0
    while start < len(text):
        end = min(start + SPAN_CHARS, len(text))
        if end < len(text):
            # Prefer a word/paragraph boundary without losing characters, including CJK.
            boundary = text.rfind(" ", start + SPAN_CHARS * 3 // 4, end)
            if boundary >= 0:
                end = boundary + 1
        value = text[start:end]
        spans.append({"start": start, "end": end, "sha256": content_hash(value), "text": value})
        if end == len(text):
            break
        start = end - SPAN_OVERLAP
    return spans


def validated_vector(value) -> list[float]:
    if not isinstance(value, list) or not value or any(isinstance(v, bool) for v in value):
        raise ValueError("Embedding must be a nonempty numeric vector")
    try:
        vector = [float(v) for v in value]
    except (ValueError, TypeError) as exc:
        raise ValueError("Embedding contains nonnumeric values") from exc
    if not all(math.isfinite(v) for v in vector):
        raise ValueError("Embedding contains nonfinite values")
    magnitude = math.sqrt(sum(v * v for v in vector))
    if not math.isfinite(magnitude) or magnitude == 0:
        raise ValueError("Embedding has zero or invalid magnitude")
    return [v / magnitude for v in vector]


def read_document(raw):
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return None


def document_is_current(raw, text: str, model: str) -> bool:
    doc = read_document(raw)
    if not isinstance(doc, dict) or doc.get("format") != FORMAT or doc.get("model") != model:
        return False
    if (doc.get("content_sha256") != content_hash(text) or doc.get("complete") is not True
            or doc.get("task_type") != "RETRIEVAL_DOCUMENT" or doc.get("char_count") != len(text)
            or doc.get("span_chars") != SPAN_CHARS or doc.get("overlap") != SPAN_OVERLAP):
        return False
    expected = text_spans(text)
    spans = doc.get("spans")
    if not isinstance(spans, list) or len(expected) != len(spans) or not spans:
        return False
    dimensions = None
    for actual, wanted in zip(spans, expected):
        if not isinstance(actual, dict) or any(actual.get(k) != wanted[k] for k in ("start", "end", "sha256")):
            return False
        try:
            vector = validated_vector(actual.get("embedding"))
        except ValueError:
            return False
        if dimensions is None:
            dimensions = len(vector)
        elif len(vector) != dimensions:
            return False
    return True


def best_match(query: list[float], raw) -> tuple[float, dict | None]:
    doc = read_document(raw)
    candidates = doc.get("spans", []) if isinstance(doc, dict) else [{"embedding": doc}]
    best_score, best_span = -1.0, None
    for span in candidates:
        if not isinstance(span, dict):
            continue
        try:
            vector = validated_vector(span.get("embedding"))
        except ValueError:
            continue
        if len(query) != len(vector):
            continue
        score = sum(a * b for a, b in zip(query, vector))
        if score > best_score:
            best_score = score
            best_span = {"start": span["start"], "end": span["end"]} if "start" in span and "end" in span else None
    return best_score, best_span


def literal_terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[^\s]+", query) if any(c.isalnum() for c in term)]


def literal_fts(query: str) -> str:
    # Every term is quoted: punctuation/operators cannot execute FTS syntax.
    return " AND ".join('"' + term.replace('"', '""') + '"' for term in literal_terms(query))


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text))
