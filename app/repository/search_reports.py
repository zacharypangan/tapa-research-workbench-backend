import csv
import re
from typing import Optional


def clean_extracted_text(text: str) -> str:
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_report_terms(query: str) -> list[str]:
    """
    Parse comma/newline/semicolon separated search terms.
    Example:
    paper mulberry, tapa cloth, tapa beater, beat
    """
    parts = re.split(r"[,;\n]+", query or "")
    terms = []

    for part in parts:
        cleaned = re.sub(r"\s+", " ", part).strip()
        if cleaned and cleaned.lower() not in {term.lower() for term in terms}:
            terms.append(cleaned)

    return terms


def build_term_pattern(term: str) -> re.Pattern:
    """
    Build a conservative regex for a term or phrase.
    Special case: beat also matches beats, beating, beaten.
    """
    cleaned = re.sub(r"\s+", " ", term.strip().lower())

    if cleaned in {"beat", "beat verb", "beat (verb)"}:
        return re.compile(r"\b(beat|beats|beating|beaten)\b", re.IGNORECASE)

    escaped_parts = [re.escape(part) for part in cleaned.split()]
    phrase_pattern = r"\s+".join(escaped_parts)

    return re.compile(rf"\b{phrase_pattern}\b", re.IGNORECASE)


def split_context_paragraphs(text: str) -> list[str]:
    """
    Split extracted text into paragraph-like units.
    Falls back to sentence grouping if the text has no paragraph breaks.
    """
    cleaned = clean_extracted_text(text)

    if not cleaned:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", cleaned) if p.strip()]

    if len(paragraphs) > 1:
        return paragraphs

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    grouped = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current) + len(sentence) + 1 <= 900:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                grouped.append(current)
            current = sentence

    if current:
        grouped.append(current)

    return grouped or [cleaned]


def find_term_contexts_in_segment(
    content_text: str,
    terms: list[str],
    context_window: int,
) -> list[dict]:
    """
    Return one result per paragraph containing at least one searched term.
    Includes paragraphs before and after the match.
    """
    paragraphs = split_context_paragraphs(content_text)
    patterns = [(term, build_term_pattern(term)) for term in terms]

    results = []

    for index, paragraph in enumerate(paragraphs):
        matched_terms = [
            term for term, pattern in patterns
            if pattern.search(paragraph)
        ]

        if not matched_terms:
            continue

        before_start = max(0, index - context_window)
        after_end = min(len(paragraphs), index + context_window + 1)

        before_paragraphs = paragraphs[before_start:index]
        after_paragraphs = paragraphs[index + 1:after_end]

        context_text = "\n\n".join(before_paragraphs + [paragraph] + after_paragraphs)

        terms_in_context = [
            term for term, pattern in patterns
            if pattern.search(context_text)
        ]

        results.append(
            {
                "paragraph_index": index,
                "matched_terms": matched_terms,
                "terms_in_context": terms_in_context,
                "all_terms_in_context": len(terms_in_context) == len(terms),
                "before": "\n\n".join(before_paragraphs),
                "match": paragraph,
                "after": "\n\n".join(after_paragraphs),
                "context_text": context_text,
            }
        )

    return results


DOMAIN_TERMS = [
    "tapa", "barkcloth", "bark cloth", "masi", "mahi", "siapo", "kapa",
    "ngatu", "hiapo", "paper mulberry", "broussonetia", "bast", "inner bark",
    "fibre", "fiber", "bark", "felt", "felting", "mallet", "beater",
    "beaters", "ike", "anvil", "yatua", "tutua", "pound", "pounding",
    "hammer", "tutumahi", "i-tututu", "samusamu", "samu masi", "daluang",
    "beaten bark paper",
]
TOOL_TERMS = [
    "beater", "beaters", "mallet", "ike", "anvil", "yatua", "tutua",
    "stone beater", "hardwood beater", "barkcloth beater",
]
PRODUCTION_TERMS = [
    "beat bark", "beat bark-cloth", "beat barkcloth", "beating bark",
    "beaten masi", "beaten bark", "pounding bark", "spread fibres",
    "spread fibers", "felting", "inner-bark", "inner bark", "bast",
]
BARK_PAPER_TERMS = ["beaten bark paper", "daluang", "amate", "fuya"]
NEGATIVE_USAGE_PATTERNS = [
    "nothing beats", "can't beat", "cannot beat", "beat these guys",
    "beat their adversaries", "beating and harassing", "beat octopus",
    "beat the lali", "drums were beaten", "lali gong", "political",
    "election", "tourist slogan",
]
RITUAL_NON_BARKCLOTH_TERMS = ["beat the lali", "lali gong", "drums were beaten", "beat octopus"]


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _term_near_domain(text: str, terms: list[str], window: int = 120) -> bool:
    lowered = text.lower()
    domain_pattern = re.compile("|".join(re.escape(term) for term in DOMAIN_TERMS), re.IGNORECASE)
    for term in terms:
        for match in re.finditer(re.escape(term), lowered, re.IGNORECASE):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            if domain_pattern.search(text[start:end]):
                return True
    return False


def _looks_like_wordlist(text: str, source_kind: str = "", source_locator: str = "") -> bool:
    lowered = f"{source_kind} {source_locator} {text[:1000]}".lower()
    has_headers = (
        "language name" in lowered
        and "concept" in lowered
        and "form" in lowered
    )
    has_lexical_rows = (
        "concept_group" in lowered
        or "pound_action" in lowered
        or re.search(r"\bconcept\s*,", lowered) is not None
    )
    return has_headers or has_lexical_rows or source_locator.lower().endswith(".csv")


def _looks_like_bibliography(text: str, source_kind: str = "", source_locator: str = "") -> bool:
    lowered = f"{source_kind} {source_locator} {text}".lower()
    if any(term in lowered for term in ["bibliography", "references", "works cited"]):
        return True
    citationish = len(re.findall(r"\(\d{4}[a-z]?\)|\b\d{4}\.", text)) >= 3
    titleish = len(re.findall(r"\b(journal|press|university|proceedings|bulletin)\b", lowered)) >= 2
    return citationish and titleish and len(text) < 2500


def maybe_parse_wordlist_rows(
    text: str,
    query_terms: Optional[list[str]] = None,
) -> list[dict]:
    lines = [line for line in (text or "").splitlines() if "," in line]
    if len(lines) < 2:
        return []

    try:
        reader = csv.DictReader(lines)
        headers = {re.sub(r"[^a-z0-9]+", "_", (field or "").strip().lower()).strip("_") for field in (reader.fieldnames or [])}
        required = {"language_name", "concept", "concept_group", "form", "source"}
        if len(required.intersection(headers)) < 3:
            return []

        rows = []
        query_patterns = [build_term_pattern(term) for term in (query_terms or [])]
        for raw_row in reader:
            normalized = {
                re.sub(r"[^a-z0-9]+", "_", (key or "").strip().lower()).strip("_"): value
                for key, value in raw_row.items()
            }
            searchable = " ".join(
                str(normalized.get(key) or "")
                for key in ["concept", "concept_group", "form"]
            )
            if query_patterns and not any(pattern.search(searchable) for pattern in query_patterns):
                continue
            rows.append(
                {
                    "language_name": normalized.get("language_name"),
                    "concept": normalized.get("concept"),
                    "concept_group": normalized.get("concept_group"),
                    "form": normalized.get("form"),
                    "source": normalized.get("source"),
                    "latitude": normalized.get("latitude"),
                    "longitude": normalized.get("longitude"),
                }
            )
            if len(rows) >= 20:
                break
        return rows
    except csv.Error:
        return []


def compute_domain_relevance_score(
    text: str,
    query_terms: list[str],
    source_kind: str = "",
    source_locator: str = "",
) -> float:
    text = text or ""
    score = 0.0
    lowered = text.lower()

    exact_matches = sum(1 for term in query_terms if build_term_pattern(term).search(text))
    score += min(exact_matches * 1.5, 4.0)

    domain_hits = sum(1 for term in DOMAIN_TERMS if term in lowered)
    score += min(domain_hits * 0.45, 4.0)

    if _term_near_domain(text, query_terms):
        score += 2.0
    if _contains_any(text, TOOL_TERMS):
        score += 1.5
    if _contains_any(text, PRODUCTION_TERMS):
        score += 1.8
    if _looks_like_wordlist(text, source_kind, source_locator):
        score += 1.5
    if _contains_any(text, NEGATIVE_USAGE_PATTERNS):
        score -= 2.5
    if _looks_like_bibliography(text, source_kind, source_locator):
        score -= 1.5

    return round(max(0.0, min(score, 10.0)), 3)


def classify_evidence_sense(
    text: str,
    query_terms: list[str],
    source_kind: str = "",
    source_locator: str = "",
) -> dict:
    text = text or ""
    score = compute_domain_relevance_score(text, query_terms, source_kind, source_locator)
    has_domain_proximity = _term_near_domain(text, query_terms)
    has_domain = _contains_any(text, DOMAIN_TERMS)
    has_negative = _contains_any(text, NEGATIVE_USAGE_PATTERNS)

    if _looks_like_wordlist(text, source_kind, source_locator):
        lexical_high = _contains_any(text, ["pound_action", "tool", "beater", "mallet", "pound"])
        return {
            "sense": "lexical_wordlist",
            "research_relevance": "high" if lexical_high or score >= 3 else "medium",
            "relevance_reason": "Looks like lexical table data with concept/form fields related to the query.",
            "evidence_type": "lexical_row_or_table",
        }

    if _looks_like_bibliography(text, source_kind, source_locator):
        return {
            "sense": "bibliographic_lead",
            "research_relevance": "medium" if has_domain else "low",
            "relevance_reason": "Looks mainly like a reference or literature-list entry.",
            "evidence_type": "bibliographic_reference",
        }

    if _contains_any(text, BARK_PAPER_TERMS):
        return {
            "sense": "bark_paper_or_related_technology",
            "research_relevance": "high" if has_domain_proximity or score >= 5 else "medium",
            "relevance_reason": "Mentions bark paper or related beaten-bark technology.",
            "evidence_type": "technology_context",
        }

    if _contains_any(text, TOOL_TERMS) and (has_domain_proximity or has_domain):
        return {
            "sense": "tool_or_beater",
            "research_relevance": "high",
            "relevance_reason": "Tool terms occur with barkcloth/tapa/masi or related production vocabulary.",
            "evidence_type": "tool_evidence",
        }

    if _contains_any(text, PRODUCTION_TERMS) and (has_domain_proximity or has_domain):
        return {
            "sense": "production_action",
            "research_relevance": "high",
            "relevance_reason": "Production action terms occur with bark, fibre, barkcloth, tapa, or related vocabulary.",
            "evidence_type": "production_process",
        }

    if _contains_any(text, RITUAL_NON_BARKCLOTH_TERMS):
        return {
            "sense": "ritual_or_non_barkcloth_action",
            "research_relevance": "medium" if has_domain else "low",
            "relevance_reason": "The query appears to describe sound, ritual, food, or other non-barkcloth action.",
            "evidence_type": "non_barkcloth_context",
        }

    if has_negative:
        return {
            "sense": "metaphor_or_general_usage",
            "research_relevance": "low",
            "relevance_reason": "General or metaphorical usage pattern appears rather than a barkcloth production action.",
            "evidence_type": "general_language_context",
        }

    return {
        "sense": "needs_review",
        "research_relevance": "medium" if has_domain or score >= 3 else "low",
        "relevance_reason": "Rule-based classifier found limited or ambiguous domain proximity; human review recommended.",
        "evidence_type": "ambiguous_context",
    }


def enrich_evidence_classification(
    item: dict,
    text: str,
    query_terms: list[str],
    source_kind: str = "",
    source_locator: str = "",
) -> dict:
    classification = classify_evidence_sense(text, query_terms, source_kind, source_locator)
    return {
        **item,
        **classification,
        "domain_relevance_score": compute_domain_relevance_score(
            text,
            query_terms,
            source_kind,
            source_locator,
        ),
    }


def matched_terms_for_text(text: str, terms: list[str]) -> list[str]:
    return [
        term for term in terms
        if build_term_pattern(term).search(text or "")
    ]
