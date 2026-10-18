"""Safe FTS5 literals and a two-character CJK substring fallback.

Existing FTS tables/triggers are reused. No tokenizer or database migration is
needed. Fallback is intentionally filtered by material before its small scan.
"""

import sqlite3

from app.repository.embedding_index import contains_cjk, literal_fts, literal_terms


def search_rows(con, query: str, material_id=None, limit=50, advanced=False) -> list[dict]:
    query = query.strip()
    if len(query) < 2:
        raise ValueError("Search requires at least two non-whitespace characters.")
    expression = query if advanced else literal_fts(query)
    if not expression:
        return []
    restriction = "AND es.source_kind NOT GLOB 'historical_*'" + (" AND es.material_id = ?" if material_id else "")
    params = [expression] + ([material_id] if material_id else []) + [limit]
    try:
        rows = con.execute(f"""SELECT es.id, es.material_id, m.title AS material_title,
            m.authors AS material_authors, m.year AS material_year,
            es.source_kind, es.source_locator, es.page_ref, es.page_index, es.content_text,
            snippet(extracted_segments_fts, 0, '[', ']', ' … ', 16) AS snippet_text,
            bm25(extracted_segments_fts) AS lexical_score
            FROM extracted_segments_fts
            JOIN extracted_segments es ON es.id = extracted_segments_fts.rowid
            JOIN materials m ON m.id = es.material_id
            WHERE extracted_segments_fts MATCH ? {restriction}
            ORDER BY bm25(extracted_segments_fts), es.id LIMIT ?""", params).fetchall()
    except sqlite3.OperationalError as exc:
        # Validation errors become a clear client response, never a 500.
        if advanced and any(term in str(exc).lower() for term in ("fts5", "unterminated", "syntax", "no such column")):
            raise ValueError("Invalid advanced FTS expression. Use literal mode or quote phrases correctly.") from exc
        raise
    result = [{**dict(row), "lexical_basis": "fts5_bm25"} for row in rows]
    terms = literal_terms(query)
    if not advanced and any(contains_cjk(term) for term in terms):
        # unicode61 keeps contiguous CJK strings as one token. Substring matching
        # restores short-term recall; instr has no LIKE wildcard/operator syntax.
        clauses = ["instr(lower(es.content_text), lower(?)) > 0" for _ in terms]
        clauses.append("es.source_kind NOT GLOB 'historical_*'")
        fallback_params = list(terms)
        if material_id:
            clauses.append("es.material_id = ?")
            fallback_params.append(material_id)
        fallback_params.append(limit)
        matches = con.execute(f"""SELECT es.id, es.material_id, m.title AS material_title,
            m.authors AS material_authors, m.year AS material_year, es.source_kind,
            es.source_locator, es.page_ref, es.page_index, es.content_text
            FROM extracted_segments es JOIN materials m ON m.id = es.material_id
            WHERE {' AND '.join(clauses)} ORDER BY es.id LIMIT ?""", fallback_params).fetchall()
        seen = {row["id"] for row in result}
        for row in matches:
            if row["id"] in seen:
                continue
            value = dict(row)
            term = next((term for term in terms if contains_cjk(term)), terms[0])
            position = value["content_text"].lower().find(term.lower())
            start = max(0, position - 80)
            excerpt = value["content_text"][start:start + 240]
            value.update(snippet_text=excerpt.replace(term, f"[{term}]"), lexical_score=None,
                         lexical_basis="cjk_literal_substring")
            result.append(value)
    return result[:limit]
