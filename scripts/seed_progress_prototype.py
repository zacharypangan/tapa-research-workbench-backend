#!/usr/bin/env python3
"""Populate the local Progress Portal with prototype-ready records and narrative arcs."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.progress.database import Base, ENGINE, SessionLocal  # noqa: E402
from app.progress.models import (  # noqa: E402
    ArcLink,
    AuditEvent,
    Concept,
    EvidenceCitation,
    Meeting,
    NarrativeArc,
    RecordConcept,
    RecordEvidence,
    ResearchRecord,
)


NAMESPACE = uuid.UUID("2f82beec-9da3-4b16-9225-e1fa0cefd723")
ACTOR_ID = "prototype-seed"


MEETING_SUMMARIES = {
    "2025-03-05": {
        "summary_en": (
            "The earliest planning frame positioned the Progress Portal as a fourth workspace "
            "inside the existing Research Workbench and rejected a CRUD-first build in favor of "
            "vertical slices grounded in real meetings."
        ),
        "summary_ja": (
            "最初期の計画では、Progress Portal を既存の Research Workbench の第4ワークスペースとして位置づけ、"
            "CRUD 先行ではなく、実際の研究会に基づく垂直スライス開発を採用する方針が定まった。"
        ),
    },
    "2025-04-14": {
        "summary_en": (
            "Security and provenance requirements became explicit: Clerk-backed identity, Railway "
            "PostgreSQL, role permissions, protected file access, and immutable evidence handling."
        ),
        "summary_ja": (
            "この段階で、Clerk による認証、Railway PostgreSQL、権限制御、保護されたファイルアクセス、"
            "不変な証拠管理といった安全性・来歴要件が明確化された。"
        ),
    },
    "2025-04-21": {
        "summary_en": (
            "The evidence model matured into one canonical research-record lifecycle with many-to-many "
            "citations, translation revision tracking, and temporal distinctions such as collection year "
            "versus publication date."
        ),
        "summary_ja": (
            "証拠モデルは、正典的な研究レコード・ライフサイクル、多対多の証拠引用、翻訳改訂履歴、"
            "採録年と発行年の区別といった時間情報の整理を含む形に発展した。"
        ),
    },
    "2025-04-27": {
        "summary_en": (
            "The 2025-04-27 pilot defined the portal's breadth: archive ingestion, repository-safe linking, "
            "unsupported-file preservation, and validation against a real multilingual meeting inventory."
        ),
        "summary_ja": (
            "2025-04-27 のパイロットは、アーカイブ取り込み、Repository 安全なリンク付け、"
            "未対応ファイルの保持、実会議資料による検証を通じて、ポータルの適用範囲を定義した。"
        ),
    },
    "2025-08-31": {
        "summary_en": (
            "Attention shifted to reviewer workflow: members should be able to convert transcript or slide "
            "evidence into draft records without copying source IDs, while reviewers govern acceptance."
        ),
        "summary_ja": (
            "この時点ではレビュー運用が中心となり、メンバーは出典 ID を手入力せずに証拠から候補レコードを作成し、"
            "レビュアーが受理を統治する設計が重視された。"
        ),
    },
    "2025-10-19": {
        "summary_en": (
            "Bilingual access and retrieval became first-class concerns: original-first ranking, Japanese/English "
            "concept aliases, and exact source locators were treated as essential for scholarly use."
        ),
        "summary_ja": (
            "日英バイリンガル検索は中核機能となり、原文優先の順位付け、概念エイリアス、"
            "厳密な出典ロケータが学術利用の必須条件として整理された。"
        ),
    },
    "2025-12-08": {
        "summary_en": (
            "Alignment work expanded the prototype from storage toward interpretation by keeping presentation-to-material "
            "links editable, confidence-scored, and protected from accidental overwrite during re-extraction."
        ),
        "summary_ja": (
            "アライメント作業により、試作システムは保存段階から解釈段階へ進み、発表資料と素材のリンクを"
            "編集可能・信頼度付き・再抽出耐性ありの状態で維持する方針が示された。"
        ),
    },
    "2026-03-19": {
        "summary_en": (
            "AI assistance was scoped carefully: structured extraction may help create candidates, but provenance, "
            "confidence, reviewer decisions, and evidence must remain visible and auditable."
        ),
        "summary_ja": (
            "AI 支援は慎重に位置づけられ、構造化抽出は候補作成を支援しても、来歴、信頼度、"
            "レビュアー判断、証拠は常に可視かつ監査可能でなければならないと整理された。"
        ),
    },
    "2026-06-20": {
        "summary_en": (
            "The maturity check emphasized narrative arcs, framework views, export regeneration, backup and restore, "
            "job recovery, and a release story that can be demonstrated end to end."
        ),
        "summary_ja": (
            "成熟度確認では、ナラティブ・アーク、フレームワーク表示、エクスポート再生成、"
            "バックアップと復元、ジョブ回復、そして端から端まで示せる公開準備が重視された。"
        ),
    },
}


RECORDS = [
    {
        "key": "workspace-foundation",
        "meeting_date": "2025-03-05",
        "record_type": "decision",
        "title_original": "Build the portal as a fourth workspace inside the existing Research Workbench",
        "title_ja": "Portal を既存 Research Workbench の第4ワークスペースとして構築する",
        "summary_en": "Preserve the React 19/Vite frontend and FastAPI backend architecture rather than migrating to Next.js or splitting the workbench apart.",
        "summary_ja": "React 19/Vite と FastAPI の既存構成を維持し、Next.js への移行や workbench の分断は行わない。",
        "original_evidence": "Build the portal as a fourth workspace inside the existing Research Workbench.",
        "evidence_locator": "implementation-plan:summary",
        "evidence_text": "Build the portal as a fourth workspace inside the existing Research Workbench.",
        "concepts": ["research provenance"],
        "details": {"phase": "foundation", "theme": "architecture"},
    },
    {
        "key": "vertical-slices",
        "meeting_date": "2025-03-05",
        "record_type": "method_claim",
        "title_original": "Replace CRUD-first development with vertical slices based on real meetings",
        "title_ja": "CRUD 先行ではなく、実会議に基づく垂直スライス開発へ置き換える",
        "summary_en": "The brief's model-heavy start was corrected so each phase proves value end to end against specific meetings instead of building abstract admin surfaces first.",
        "summary_ja": "モデルと CRUD 画面を先に積み上げる方針を改め、各段階で特定の会議を通じて端から端まで価値を実証する設計にした。",
        "original_evidence": "Replace the brief's build-18-models-and-CRUD-pages-first approach with end-to-end vertical slices based on real meetings.",
        "evidence_locator": "execution-plan:critical-adjustments",
        "evidence_text": "Replace the brief's build-18-models-and-CRUD-pages-first approach with end-to-end vertical slices based on real meetings.",
        "concepts": ["knowledge graph"],
        "details": {"phase": "foundation", "theme": "delivery"},
    },
    {
        "key": "auth-rbac",
        "meeting_date": "2025-04-14",
        "record_type": "decision",
        "title_original": "Protect Repository and Progress with Clerk JWT verification and organization permissions",
        "title_ja": "Clerk の JWT 検証と組織権限で Repository と Progress を保護する",
        "summary_en": "Unauthenticated access must return 401, role violations must return 403, and the same permission model should gate repository files, extractions, and progress records.",
        "summary_ja": "未認証は 401、権限不足は 403 とし、Repository のファイル・抽出物・Progress レコードを同一の権限モデルで保護する。",
        "original_evidence": "Add Clerk, authenticated API requests, JWT verification, and organization permissions progress:read, progress:write, progress:review, and progress:admin.",
        "evidence_locator": "execution-plan:phase-1",
        "evidence_text": "Add Clerk, authenticated API requests, JWT verification, and organization permissions progress:read, progress:write, progress:review, and progress:admin.",
        "concepts": ["research provenance"],
        "details": {"phase": "foundation-security", "theme": "security"},
    },
    {
        "key": "immutable-provenance",
        "meeting_date": "2025-04-14",
        "record_type": "repository_enrichment",
        "title_original": "Use immutable file hashes and protected downloads to preserve evidence integrity",
        "title_ja": "不変ハッシュと保護ダウンロードで証拠の完全性を守る",
        "summary_en": "Repository originals remain immutable; replacements create new file versions and checksums rather than overwriting prior evidence.",
        "summary_ja": "Repository の原本は不変とし、差し替えは上書きではなく新しいファイル版とチェックサムを生成する。",
        "original_evidence": "Add authentication, audit history, immutable file hashes, and protected downloads.",
        "evidence_locator": "execution-plan:critical-adjustments",
        "evidence_text": "Add authentication, audit history, immutable file hashes, and protected downloads.",
        "concepts": ["research provenance", "optical character recognition"],
        "details": {"phase": "foundation-security", "theme": "repository"},
    },
    {
        "key": "canonical-ledger",
        "meeting_date": "2025-04-21",
        "record_type": "method_claim",
        "title_original": "Use one canonical research-record lifecycle with subtype detail tables",
        "title_ja": "サブタイプ詳細表を伴う単一の正典研究レコード・ライフサイクルを採用する",
        "summary_en": "Generic and specialized records must not drift apart; one shared lifecycle keeps review state, provenance, and evidence consistent across the portal.",
        "summary_ja": "汎用レコードと特化レコードの乖離を防ぐため、共通ライフサイクルでレビュー状態・来歴・証拠を一貫管理する。",
        "original_evidence": "Use one canonical research-record lifecycle with subtype detail tables, preventing generic and specialized records from drifting apart.",
        "evidence_locator": "execution-plan:critical-adjustments",
        "evidence_text": "Use one canonical research-record lifecycle with subtype detail tables, preventing generic and specialized records from drifting apart.",
        "concepts": ["research provenance"],
        "details": {"phase": "core-ledger", "theme": "data-model"},
    },
    {
        "key": "collection-year-distinction",
        "meeting_date": "2025-04-21",
        "record_type": "temporal_issue",
        "title_original": "Collection year must be separated from publication date",
        "title_ja": "採録年と発行年は区別して扱わなければならない",
        "summary_en": "Temporal evidence becomes misleading when collection year, publication year, and processing year collapse into one field; the portal needs explicit distinctions and concept aliases.",
        "summary_ja": "採録年・発行年・処理年が一つの欄に混在すると時間情報が歪むため、明示的な区別と概念エイリアスが必要である。",
        "original_evidence": "Rank original exact matches above translations, aliases, and fuzzy matches; collection year and 時間情報 should retrieve the same concept cluster.",
        "evidence_locator": "execution-plan:phase-5",
        "evidence_text": "Collection year and 時間情報 should retrieve the same seeded concept cluster while displaying original evidence first.",
        "concepts": ["collection year", "publication date", "conceptualized time"],
        "details": {"phase": "core-ledger", "theme": "temporal-reasoning"},
    },
    {
        "key": "pilot-breadth",
        "meeting_date": "2025-04-27",
        "record_type": "action_item",
        "title_original": "Validate breadth with the 2025-04-27 pilot archive before expanding",
        "title_ja": "拡張前に 2025-04-27 パイロット・アーカイブで適用範囲を検証する",
        "summary_en": "The pilot is the breadth check: importer behavior, protected files, transcript handling, review flow, and bilingual retrieval should all prove themselves on the 2025-04-27 meeting first.",
        "summary_ja": "パイロットは適用範囲確認であり、インポータ、保護ファイル、文字起こし、レビュー、バイリンガル検索をまず 2025-04-27 会議で実証する。",
        "original_evidence": "Validate first with the 2025-04-27 meeting, then use 2026-06-20 as the maturity check.",
        "evidence_locator": "execution-plan:summary",
        "evidence_text": "Validate first with the 2025-04-27 meeting, then use 2026-06-20 as the maturity check.",
        "concepts": ["research provenance"],
        "details": {"phase": "pilot", "theme": "acceptance"},
    },
    {
        "key": "idempotent-importer",
        "meeting_date": "2025-04-27",
        "record_type": "repository_enrichment",
        "title_original": "Import the archive idempotently and preserve unsupported assets instead of hiding them",
        "title_ja": "アーカイブは冪等に取り込み、未対応ファイルも隠さず保持する",
        "summary_en": "SHA-256 hashes, import provenance, and parser status allow the portal to rerun ingestion without duplicates while showing DOCX, XLSX, PPSX, PPTM, and MP4 items as preserved review targets.",
        "summary_ja": "SHA-256、取り込み来歴、パーサ状態により重複なく再実行でき、DOCX・XLSX・PPSX・PPTM・MP4 も保存対象として可視化できる。",
        "original_evidence": "Unsupported DOCX/XLSX/PPSX/PPTM files are preserved and visibly marked rather than silently misparsed.",
        "evidence_locator": "execution-plan:phase-3",
        "evidence_text": "Unsupported DOCX/XLSX/PPSX/PPTM files are preserved and visibly marked rather than silently misparsed.",
        "concepts": ["optical character recognition", "research provenance"],
        "details": {"phase": "pilot", "theme": "ingestion"},
    },
    {
        "key": "reviewer-workflow",
        "meeting_date": "2025-08-31",
        "record_type": "decision",
        "title_original": "Reviewers should be able to accept evidence-linked records without manual source-ID copying",
        "title_ja": "レビュアーは出典 ID を手入力せずに証拠付きレコードを受理できるべきである",
        "summary_en": "The meeting-centered workspace should let members draft from transcript cues or slide evidence while preserving exact repository locators for reviewers.",
        "summary_ja": "会議中心ワークスペースでは、メンバーが文字起こしやスライド証拠から候補を作成し、正確な Repository ロケータを保持したままレビュアーへ渡せる必要がある。",
        "original_evidence": "A reviewer can turn a transcript or slide selection into an evidence-linked record without manually copying source identifiers.",
        "evidence_locator": "execution-plan:phase-4",
        "evidence_text": "A reviewer can turn a transcript or slide selection into an evidence-linked record without manually copying source identifiers.",
        "concepts": ["research provenance"],
        "details": {"phase": "review-workflow", "theme": "review"},
    },
    {
        "key": "meeting-centered-ui",
        "meeting_date": "2025-08-31",
        "record_type": "technique",
        "title_original": "Use a meeting-centered workspace to unify metadata, presentations, transcript cues, and review",
        "title_ja": "会議中心ワークスペースでメタデータ、発表、文字起こし、レビューを統合する",
        "summary_en": "Progress, decisions, actions, questions, techniques, and review queue belong in one meeting view so the user can follow evidence without switching systems.",
        "summary_ja": "進捗、決定、アクション、問い、技法、レビュー待ちを一つの会議画面に集約し、証拠の流れを切らさないようにする。",
        "original_evidence": "Build the meeting-centered interface: metadata, sessions, presentations, materials, transcript timeline, slide/page evidence, progress, decisions, actions, questions, techniques, and review queue.",
        "evidence_locator": "execution-plan:phase-4",
        "evidence_text": "Build the meeting-centered interface: metadata, sessions, presentations, materials, transcript timeline, slide/page evidence, progress, decisions, actions, questions, techniques, and review queue.",
        "concepts": ["knowledge graph"],
        "details": {"phase": "review-workflow", "theme": "ui"},
    },
    {
        "key": "bilingual-search",
        "meeting_date": "2025-10-19",
        "record_type": "technique",
        "title_original": "Rank original evidence first while expanding bilingual search through concept aliases",
        "title_ja": "概念エイリアスで拡張しつつ、原文証拠を最優先に順位付けする",
        "summary_en": "Japanese and English retrieval should converge through aliases such as collection year and 時間情報, but the UI must still foreground original evidence and exact locators.",
        "summary_ja": "collection year と 時間情報のようなエイリアスで日英検索を接続しつつ、UI では原文証拠と厳密なロケータを最優先表示する。",
        "original_evidence": "Add original-first displays, Japanese/English toggles, concept aliases, and cross-language query expansion.",
        "evidence_locator": "execution-plan:phase-5",
        "evidence_text": "Add original-first displays, Japanese/English toggles, concept aliases, and cross-language query expansion.",
        "concepts": ["collection year", "conceptualized time", "research provenance"],
        "details": {"phase": "bilingual-search", "theme": "search"},
    },
    {
        "key": "time-info-search",
        "meeting_date": "2025-10-19",
        "record_type": "temporal_issue",
        "title_original": "Time information queries should resolve to the same conceptual cluster as collection year",
        "title_ja": "時間情報の検索は collection year と同じ概念クラスタに到達すべきである",
        "summary_en": "The portal needs bilingual concept normalization so temporal questions remain semantically stable across languages and source traditions.",
        "summary_ja": "時間に関する問いが言語や資料系列を超えて安定して検索できるよう、バイリンガル概念正規化が必要である。",
        "original_evidence": "Collection year and 時間情報 retrieve the same seeded concept cluster while displaying original evidence first.",
        "evidence_locator": "execution-plan:phase-5-gate",
        "evidence_text": "Collection year and 時間情報 retrieve the same seeded concept cluster while displaying original evidence first.",
        "concepts": ["collection year", "conceptualized time"],
        "details": {"phase": "bilingual-search", "theme": "temporal-search"},
    },
    {
        "key": "alignment-confidence",
        "meeting_date": "2025-12-08",
        "record_type": "spatial_issue",
        "title_original": "Presentation-to-material alignment must stay editable and confidence-scored",
        "title_ja": "発表と素材のアライメントは編集可能かつ信頼度付きで維持する",
        "summary_en": "Program order, presenter names, timestamps, titles, and repeated terms can assist alignment, but the resulting links must remain editable and explicitly scored.",
        "summary_ja": "プログラム順、発表者名、時刻、タイトル、反復語はアライメントを支援できるが、生成されたリンクは編集可能で信頼度が明示されていなければならない。",
        "original_evidence": "Add manual and assisted presentation-to-material and transcript-to-slide alignment using program order, presenter names, timestamps, titles, and repeated terms.",
        "evidence_locator": "execution-plan:phase-6",
        "evidence_text": "Add manual and assisted presentation-to-material and transcript-to-slide alignment using program order, presenter names, timestamps, titles, and repeated terms.",
        "concepts": ["conceptualized space", "geographic information system", "spatial uncertainty"],
        "details": {"phase": "alignment", "theme": "alignment"},
    },
    {
        "key": "alignment-preservation",
        "meeting_date": "2025-12-08",
        "record_type": "uncertainty_issue",
        "title_original": "Re-extraction must not overwrite human alignment decisions",
        "title_ja": "再抽出は人手によるアライメント判断を上書きしてはならない",
        "summary_en": "Alignment links need method and confidence metadata and must survive re-extraction without being replaced by automated guesses.",
        "summary_ja": "アライメントには手法・信頼度メタデータが必要であり、再抽出時にも自動推定で置き換えられてはならない。",
        "original_evidence": "Links remain editable, carry confidence/method metadata, and survive re-extraction without being overwritten.",
        "evidence_locator": "execution-plan:phase-6-gate",
        "evidence_text": "Links remain editable, carry confidence/method metadata, and survive re-extraction without being overwritten.",
        "concepts": ["spatial uncertainty", "research provenance"],
        "details": {"phase": "alignment", "theme": "durability"},
    },
    {
        "key": "ai-candidates-only",
        "meeting_date": "2026-03-19",
        "record_type": "technique",
        "title_original": "AI extraction may draft candidates but may not accept records or alter original evidence",
        "title_ja": "AI 抽出は候補作成を支援しても、レコード受理や原証拠改変は許されない",
        "summary_en": "AI output always enters candidate or needs_review state, and every extraction run must preserve prompt version, model, evidence source, and confidence.",
        "summary_ja": "AI 出力は常に candidate か needs_review に留まり、各抽出実行は prompt version、model、証拠源、confidence を保持する。",
        "original_evidence": "AI output always enters candidate or needs_review; no model action may accept records or alter original evidence.",
        "evidence_locator": "execution-plan:phase-7",
        "evidence_text": "AI output always enters candidate or needs_review; no model action may accept records or alter original evidence.",
        "concepts": ["research provenance"],
        "details": {"phase": "ai-candidates", "theme": "ai-governance"},
    },
    {
        "key": "ai-provenance",
        "meeting_date": "2026-03-19",
        "record_type": "uncertainty_issue",
        "title_original": "Model version, prompt version, reviewer decision, and confidence are part of the evidence chain",
        "title_ja": "モデル版、プロンプト版、レビュアー判断、信頼度も証拠連鎖の一部である",
        "summary_en": "AI assistance becomes governable only when provenance metadata is treated as first-class evidence rather than hidden implementation detail.",
        "summary_ja": "AI 支援を統治可能にするには、来歴メタデータを隠れた実装詳細ではなく一次的な証拠として扱う必要がある。",
        "original_evidence": "Treat AI model, prompt version, source evidence, confidence, and reviewer decision as provenance.",
        "evidence_locator": "execution-plan:critical-adjustments",
        "evidence_text": "Treat AI model, prompt version, source evidence, confidence, and reviewer decision as provenance.",
        "concepts": ["research provenance", "spatial uncertainty"],
        "details": {"phase": "ai-candidates", "theme": "provenance"},
    },
    {
        "key": "narrative-views",
        "meeting_date": "2026-06-20",
        "record_type": "decision",
        "title_original": "Narrative arcs, frameworks, and reports must regenerate from stored evidence after deployment",
        "title_ja": "ナラティブ・アーク、フレームワーク、レポートは配備後も保存済み証拠から再生成できなければならない",
        "summary_en": "Meeting, member, decision, framework, repository-enrichment, and narrative reports should be deterministic products of the stored ledger, not manual slideware.",
        "summary_ja": "会議、メンバー、決定、フレームワーク、Repository 拡充、ナラティブの各レポートは、手作業ではなく保存済みレジャーから決定論的に再生成されるべきである。",
        "original_evidence": "Meeting, member, decision, framework, repository-enrichment, and narrative reports regenerate from stored evidence after a clean deployment.",
        "evidence_locator": "execution-plan:phase-8-gate",
        "evidence_text": "Meeting, member, decision, framework, repository-enrichment, and narrative reports regenerate from stored evidence after a clean deployment.",
        "concepts": ["knowledge graph", "research provenance"],
        "details": {"phase": "release-hardening", "theme": "narrative"},
    },
    {
        "key": "release-hardening",
        "meeting_date": "2026-06-20",
        "record_type": "action_item",
        "title_original": "Release hardening requires backups, restore verification, protected-file tests, and job recovery",
        "title_ja": "公開前の堅牢化にはバックアップ、復元検証、保護ファイル試験、ジョブ回復が必要である",
        "summary_en": "A showable prototype still needs operational evidence: smoke tests, backup and restore, export checks, protected downloads, and recovery from interrupted jobs.",
        "summary_ja": "見せられる試作版であっても、スモークテスト、バックアップと復元、エクスポート確認、保護ダウンロード、ジョブ中断からの回復といった運用証拠が必要である。",
        "original_evidence": "Complete deployment smoke tests, database backups and restore verification, job recovery, audit/export checks, and protected-file tests.",
        "evidence_locator": "execution-plan:phase-8",
        "evidence_text": "Complete deployment smoke tests, database backups and restore verification, job recovery, audit/export checks, and protected-file tests.",
        "concepts": ["research provenance"],
        "details": {"phase": "release-hardening", "theme": "operations"},
    },
]


ARCS = [
    {
        "key": "archive-to-ledger",
        "title_en": "From Archive Inventory to Evidence Ledger",
        "title_ja": "アーカイブ在庫から証拠レジャーへ",
        "description_en": "This arc follows the portal's core transformation: existing files become protected repository assets, then evidence-linked records, then reproducible reports.",
        "description_ja": "このアークは、既存ファイルが保護された Repository 資産となり、証拠付きレコードとなり、最終的に再生成可能なレポートへ至る流れをたどる。",
        "links": [
            ("workspace-foundation", "origin", "The portal is embedded inside the workbench rather than built as a detached admin app.", "Portal は分離した管理画面ではなく workbench の内部に埋め込まれる。"),
            ("canonical-ledger", "refinement", "One canonical record lifecycle turns scattered notes into governed research records.", "単一のレコード・ライフサイクルが散在メモを統治された研究レコードへ変換する。"),
            ("pilot-breadth", "problem", "The 2025-04-27 archive is the first real test of whether the ledger can hold multilingual meeting evidence.", "2025-04-27 アーカイブは、多言語会議証拠を保持できるかを試す最初の実地検証である。"),
            ("idempotent-importer", "evidence", "Idempotent import and preserved-only visibility keep the archive honest during ingestion.", "冪等インポートと preserved-only 可視化により、取り込み時の誠実さを保つ。"),
            ("reviewer-workflow", "decision", "Reviewers can accept records directly from evidence instead of reconstructing provenance by hand.", "レビュアーは来歴を手で再構成せずに、証拠から直接レコードを受理できる。"),
            ("narrative-views", "follow_up", "Stored evidence can now regenerate the narrative story of the project.", "保存済み証拠からプロジェクトの物語を再生成できるようになる。"),
        ],
    },
    {
        "key": "bilingual-temporal-reasoning",
        "title_en": "Bilingual Time and Alignment Reasoning",
        "title_ja": "バイリンガルな時間推論とアライメント",
        "description_en": "This arc develops the portal's interpretive layer: temporal distinctions, bilingual search, and alignment confidence are treated as scholarly necessities rather than UI polish.",
        "description_ja": "このアークは、時間区別、バイリンガル検索、アライメント信頼度を UI の飾りではなく学術的必須条件として扱う解釈層の発展を示す。",
        "links": [
            ("collection-year-distinction", "origin", "Temporal modeling starts with distinguishing collection year from publication date.", "時間モデリングは採録年と発行年の区別から始まる。"),
            ("bilingual-search", "refinement", "Original-first bilingual retrieval turns that distinction into a usable scholarly workflow.", "原文優先のバイリンガル検索により、その区別が実用的な学術ワークフローになる。"),
            ("time-info-search", "decision", "Collection year and 時間情報 must meet in the same concept cluster.", "collection year と 時間情報は同じ概念クラスタで出会わなければならない。"),
            ("alignment-confidence", "evidence", "Alignment adds confidence-bearing links between presentations, materials, and transcripts.", "アライメントは発表・素材・文字起こしの間に信頼度付きリンクを追加する。"),
            ("alignment-preservation", "follow_up", "Those links must survive re-extraction without losing human judgment.", "そのリンクは人間の判断を失わずに再抽出を生き残る必要がある。"),
            ("narrative-views", "follow_up", "Framework and narrative exports make temporal and spatial interpretation demonstrable.", "フレームワークとナラティブのエクスポートにより、時間・空間の解釈を実演可能にする。"),
        ],
    },
    {
        "key": "governed-ai-assistance",
        "title_en": "Governed AI Assistance",
        "title_ja": "統治された AI 支援",
        "description_en": "This arc captures the portal's stance toward automation: secure, provenance-aware, reviewer-governed assistance that never outruns the evidence.",
        "description_ja": "このアークは、自動化に対するポータルの姿勢を示す。すなわち、安全で来歴を保持し、レビュアーに統治され、証拠を超走しない AI 支援である。",
        "links": [
            ("auth-rbac", "origin", "Security and role boundaries are prerequisites for any assisted workflow.", "安全性と役割境界は、あらゆる支援ワークフローの前提条件である。"),
            ("immutable-provenance", "evidence", "Immutable hashes and protected files establish trust in the underlying corpus.", "不変ハッシュと保護ファイルが基盤コーパスへの信頼を確立する。"),
            ("ai-candidates-only", "problem", "AI must stop at candidate generation rather than accepting records on its own.", "AI は候補生成で止まり、自らレコードを受理してはならない。"),
            ("ai-provenance", "decision", "Prompt version, model, evidence, confidence, and reviewer judgment become first-class provenance.", "プロンプト版、モデル、証拠、信頼度、レビュアー判断は一次的な来歴となる。"),
            ("release-hardening", "follow_up", "Operational checks complete the trust story by proving recovery, export, and protection workflows.", "回復・エクスポート・保護ワークフローを証明する運用確認が、信頼の物語を完成させる。"),
        ],
    },
]


def stable_uuid(kind: str, key: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{kind}:{key}"))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def upsert_record(session, meeting_by_date, concept_by_label, payload):
    record_id = stable_uuid("record", payload["key"])
    evidence_id = stable_uuid("evidence", payload["key"])
    meeting = meeting_by_date[payload["meeting_date"]]
    now = utc_now()

    record = session.get(ResearchRecord, record_id)
    record_data = {
        "id": record_id,
        "record_type": payload["record_type"],
        "title_original": payload["title_original"],
        "title_en": payload["title_original"],
        "title_ja": payload["title_ja"],
        "summary_en": payload["summary_en"],
        "summary_ja": payload["summary_ja"],
        "original_evidence": payload["original_evidence"],
        "original_language": "en",
        "meeting_id": meeting.id,
        "status": "accepted",
        "confidence": 0.98,
        "details": {
            **payload["details"],
            "prototype_seed_key": payload["key"],
            "prototype_story": True,
        },
        "created_by": ACTOR_ID,
        "reviewed_by": ACTOR_ID,
        "reviewed_at": now,
        "notes": "Prototype seed generated from the implementation brief and execution plan.",
    }
    if record:
        for field, value in record_data.items():
            setattr(record, field, value)
    else:
        record = ResearchRecord(**record_data)
        session.add(record)
        session.flush()
        session.add(
            AuditEvent(
                entity_type="research_record",
                entity_id=record.id,
                action="prototype_seeded",
                actor_id=ACTOR_ID,
                changes={"meeting_date": payload["meeting_date"], "key": payload["key"]},
            )
        )

    evidence = session.get(EvidenceCitation, evidence_id)
    evidence_data = {
        "id": evidence_id,
        "source_kind": "note",
        "source_locator": payload["evidence_locator"],
        "citation_text_original": payload["evidence_text"],
        "citation_text_en": payload["evidence_text"],
        "notes": "Evidence note seeded from the implementation plan.",
    }
    if evidence:
        for field, value in evidence_data.items():
            setattr(evidence, field, value)
    else:
        evidence = EvidenceCitation(**evidence_data)
        session.add(evidence)
        session.flush()

    link = session.get(RecordEvidence, {"record_id": record.id, "evidence_id": evidence.id})
    if not link:
        session.add(
            RecordEvidence(
                record_id=record.id,
                evidence_id=evidence.id,
                relation="supports",
                order_index=0,
            )
        )

    session.query(RecordConcept).filter(RecordConcept.record_id == record.id).delete()
    for label in payload["concepts"]:
        concept = concept_by_label.get(label)
        if concept:
            session.add(RecordConcept(record_id=record.id, concept_id=concept.id))
    return record.id


def upsert_arc(session, record_ids_by_key, arc_payload):
    arc_id = stable_uuid("arc", arc_payload["key"])
    arc = session.get(NarrativeArc, arc_id)
    data = {
        "id": arc_id,
        "title_en": arc_payload["title_en"],
        "title_ja": arc_payload["title_ja"],
        "description_en": arc_payload["description_en"],
        "description_ja": arc_payload["description_ja"],
        "status": "active",
        "notes": "Prototype narrative arc seeded from the original portal execution plan.",
    }
    if arc:
        for field, value in data.items():
            setattr(arc, field, value)
    else:
        arc = NarrativeArc(**data)
        session.add(arc)
        session.flush()
        session.add(
            AuditEvent(
                entity_type="narrative_arc",
                entity_id=arc.id,
                action="prototype_seeded",
                actor_id=ACTOR_ID,
                changes={"key": arc_payload["key"]},
            )
        )

    for order_index, (record_key, role, summary_en, summary_ja) in enumerate(arc_payload["links"]):
        source_id = record_ids_by_key[record_key]
        link_id = stable_uuid("arc-link", f"{arc_payload['key']}:{record_key}")
        link = session.get(ArcLink, link_id)
        link_data = {
            "id": link_id,
            "arc_id": arc.id,
            "source_kind": "research_record",
            "source_id": source_id,
            "role_in_arc": role,
            "summary_en": summary_en,
            "summary_ja": summary_ja,
            "confidence": 0.99,
            "order_index": order_index,
            "notes": "Prototype arc progression seeded from the execution plan.",
        }
        if link:
            for field, value in link_data.items():
                setattr(link, field, value)
        else:
            session.add(ArcLink(**link_data))


def main() -> int:
    Base.metadata.create_all(ENGINE)
    with SessionLocal() as session:
        meeting_by_date = {
            meeting.meeting_date.isoformat(): meeting
            for meeting in session.query(Meeting).all()
        }
        missing = sorted(set(MEETING_SUMMARIES) - set(meeting_by_date))
        if missing:
            raise SystemExit(f"Missing meetings for prototype seed: {', '.join(missing)}")

        for meeting_date, summaries in MEETING_SUMMARIES.items():
            meeting = meeting_by_date[meeting_date]
            meeting.summary_en = summaries["summary_en"]
            meeting.summary_ja = summaries["summary_ja"]

        concept_by_label = {
            concept.label_en: concept
            for concept in session.query(Concept).all()
        }

        record_ids_by_key: dict[str, str] = {}
        for payload in RECORDS:
            record_ids_by_key[payload["key"]] = upsert_record(
                session,
                meeting_by_date,
                concept_by_label,
                payload,
            )

        for arc_payload in ARCS:
            upsert_arc(session, record_ids_by_key, arc_payload)

        session.commit()

    print(f"meeting_summaries_updated: {len(MEETING_SUMMARIES)}")
    print(f"records_seeded: {len(RECORDS)}")
    print(f"narrative_arcs_seeded: {len(ARCS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
