#!/usr/bin/env python3
"""Seed the initial bilingual project thesaurus without creating duplicates."""

from __future__ import annotations

import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.progress.database import Base, ENGINE, SessionLocal  # noqa: E402
from app.progress.models import Concept  # noqa: E402


CONCEPTS = [
    {
        "label_en": "collection year",
        "label_ja": "採録年",
        "aliases_en": ["collection date", "fieldwork date", "temporal metadata", "time information"],
        "aliases_ja": ["収集年", "調査年", "時間情報", "時点", "時期"],
        "concept_type": "temporal",
    },
    {
        "label_en": "publication date",
        "label_ja": "発行年",
        "aliases_en": ["publication year", "published date"],
        "aliases_ja": ["出版年", "刊行年", "発行日"],
        "concept_type": "temporal",
    },
    {
        "label_en": "spatial uncertainty",
        "label_ja": "空間的不確実性",
        "aliases_en": ["uncertain location", "multiple candidate locations", "representative point"],
        "aliases_ja": ["位置の不確実性", "候補地点", "代表地点"],
        "concept_type": "spatial",
    },
    {
        "label_en": "conceptualized space",
        "label_ja": "概念化された空間",
        "aliases_en": ["conceptual place", "interpreted location"],
        "aliases_ja": ["概念的場所", "解釈された場所"],
        "concept_type": "spatial",
    },
    {
        "label_en": "conceptualized time",
        "label_ja": "概念化された時間",
        "aliases_en": ["interpreted period", "expert-defined period", "inferred date"],
        "aliases_ja": ["解釈された時期", "専門家定義の時期", "推定年代"],
        "concept_type": "temporal",
    },
    {
        "label_en": "geographic information system",
        "label_ja": "地理情報システム",
        "aliases_en": ["GIS", "analytical map interface", "Kepler GL"],
        "aliases_ja": ["GIS", "地理情報", "分析的地図インターフェース"],
        "concept_type": "method",
    },
    {
        "label_en": "optical character recognition",
        "label_ja": "光学文字認識",
        "aliases_en": ["OCR", "text extraction", "scanned source"],
        "aliases_ja": ["OCR", "文字抽出", "スキャン資料"],
        "concept_type": "method",
    },
    {
        "label_en": "research provenance",
        "label_ja": "研究来歴",
        "aliases_en": ["evidence link", "source traceability", "data lineage"],
        "aliases_ja": ["証拠リンク", "出典追跡", "データ来歴"],
        "concept_type": "governance",
    },
    {
        "label_en": "knowledge graph",
        "label_ja": "知識グラフ",
        "aliases_en": ["semantic graph", "entity relation", "ontology"],
        "aliases_ja": ["意味グラフ", "実体関係", "オントロジー"],
        "concept_type": "method",
    },
]


def main() -> int:
    Base.metadata.create_all(ENGINE)
    created = 0
    with SessionLocal() as session:
        for payload in CONCEPTS:
            exists = session.query(Concept).filter(Concept.label_en == payload["label_en"]).one_or_none()
            if exists:
                continue
            session.add(
                Concept(
                    **payload,
                    source_vocabulary="Spatiotemporal Linguistics Data Governance",
                    source_version="2026-07-01",
                )
            )
            created += 1
        session.commit()
    print(f"concepts_created: {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
