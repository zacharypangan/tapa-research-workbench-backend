from contextlib import nullcontext
import sqlite3
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import repository as repository_api
from app.repository.semantic_graph import (
    build_semantic_graph,
    explain_semantic_relation,
    review_semantic_candidate,
    review_semantic_relation,
    semantic_graph_payload,
)
from app.repository.semantic_rules import (
    RELATION_RULES,
    candidate_rule_id,
    primary_relation_rule_id,
    semantic_relation_rules_payload,
)


def create_source_schema(con: sqlite3.Connection):
    con.executescript(
        """
        CREATE TABLE materials (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT,
            year TEXT,
            source_type TEXT NOT NULL,
            collection TEXT,
            abstract_or_notes TEXT,
            source_url TEXT,
            language TEXT,
            region TEXT,
            uploaded_by TEXT,
            raw_reference TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            keywords TEXT,
            auto_keywords TEXT
        );
        CREATE TABLE extracted_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            source_locator TEXT NOT NULL,
            page_ref TEXT NOT NULL,
            page_index INTEGER NOT NULL,
            content_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE image_evidence (
            id TEXT PRIMARY KEY,
            material_id TEXT NOT NULL,
            evidence_type TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            source_locator TEXT NOT NULL,
            page_ref TEXT NOT NULL,
            page_index INTEGER NOT NULL,
            extraction_method TEXT NOT NULL,
            ocr_text TEXT,
            visual_caption TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE observations (
            id TEXT PRIMARY KEY,
            material_id TEXT NOT NULL,
            observation_type TEXT NOT NULL,
            observed_text TEXT NOT NULL,
            context_quote TEXT,
            notes TEXT,
            source_segment_id INTEGER,
            source_image_id TEXT,
            source_page_ref TEXT,
            source_locator TEXT,
            created_at TEXT NOT NULL
        );
        """
    )


class SemanticGraphTest(unittest.TestCase):
    def setUp(self):
        self.con = sqlite3.connect(":memory:", check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")
        create_source_schema(self.con)
        for index in range(10):
            material_id = f"material-{index + 1}"
            keywords = "barkcloth; weaving" if index < 2 else None
            auto_keywords = "machine suggestion" if index == 2 else None
            self.con.execute(
                """
                INSERT INTO materials (
                    id, title, authors, year, source_type, collection,
                    abstract_or_notes, language, region, status,
                    created_at, updated_at, keywords, auto_keywords
                )
                VALUES (?, ?, ?, '2020', 'publication', ?, ?, 'English', ?, 'metadata_complete', ?, ?, ?, ?)
                """,
                (
                    material_id,
                    f"Material {index + 1}",
                    "Researcher One",
                    "Collection A" if index < 5 else "Collection B",
                    "Barkcloth and weaving evidence.",
                    "Fiji" if index == 0 else None,
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                    keywords,
                    auto_keywords,
                ),
            )
        segment_rows = [
            ("material-1", "p. 1", "Barkcloth and weaving were recorded in 2019."),
            ("material-1", "pp. 100-101", "Barkcloth and weaving appear on pages 100-101."),
            ("material-1", "p. 2", "A comparison was made near Mystery Island."),
            ("material-2", "p. 3", "Barkcloth and weaving were discussed in 2020."),
        ]
        for page_index, (material_id, page_ref, text) in enumerate(segment_rows):
            self.con.execute(
                """
                INSERT INTO extracted_segments (
                    material_id, source_kind, source_locator, page_ref,
                    page_index, content_text, created_at
                )
                VALUES (?, 'pdf', ?, ?, ?, ?, ?)
                """,
                (
                    material_id,
                    f"file.pdf#{page_ref}",
                    page_ref,
                    page_index,
                    text,
                    "2026-01-01T00:00:00+00:00",
                ),
            )
        self.con.commit()

    def tearDown(self):
        self.con.close()

    def test_semantic_projection_is_stable_and_reviewable(self):
        first = build_semantic_graph(self.con)
        first_entity_ids = {
            row["id"] for row in self.con.execute("SELECT id FROM kg_entities").fetchall()
        }
        first_relation_ids = {
            row["id"] for row in self.con.execute("SELECT id FROM kg_relations").fetchall()
        }

        self.assertGreater(first["mention_count"], first["entity_count"])
        self.assertEqual(
            self.con.execute("SELECT COUNT(*) FROM kg_entities WHERE entity_type = 'segment'").fetchone()[0],
            0,
        )
        self.assertEqual(
            self.con.execute(
                "SELECT COUNT(*) FROM kg_entities WHERE entity_type IN ('corpus', 'root', 'whole_repository')"
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.con.execute(
                "SELECT COUNT(*) FROM kg_entities WHERE entity_type = 'time_period' AND label = '100-101'"
            ).fetchone()[0],
            0,
        )
        invalid = self.con.execute(
            "SELECT resolution_status, time_type FROM kg_time_resolution WHERE mention_label = '100-101'"
        ).fetchone()
        self.assertIsNotNone(invalid)
        self.assertEqual(invalid["resolution_status"], "invalid")
        self.assertEqual(invalid["time_type"], "invalid_page_range")

        candidate = self.con.execute(
            "SELECT * FROM kg_candidate_relations WHERE predicate = 'co_occurs_with'"
        ).fetchone()
        self.assertIsNotNone(candidate)
        self.assertGreaterEqual(candidate["evidence_count"], 3)
        self.assertEqual(candidate["document_count"], 2)
        self.assertEqual(candidate["review_status"], "needs_review")

        relation = self.con.execute(
            "SELECT * FROM kg_relations WHERE predicate = 'has_topic' ORDER BY evidence_count DESC LIMIT 1"
        ).fetchone()
        review_semantic_relation(self.con, relation["id"], "rejected")
        review_semantic_candidate(self.con, candidate["id"], "accepted")
        self.con.commit()

        build_semantic_graph(self.con)
        second_entity_ids = {
            row["id"] for row in self.con.execute("SELECT id FROM kg_entities").fetchall()
        }
        second_relation_ids = {
            row["id"] for row in self.con.execute("SELECT id FROM kg_relations").fetchall()
        }
        self.assertEqual(first_entity_ids, second_entity_ids)
        self.assertTrue(first_relation_ids.issubset(second_relation_ids))
        self.assertEqual(
            self.con.execute("SELECT assertion_status FROM kg_relations WHERE id = ?", (relation["id"],)).fetchone()[0],
            "rejected",
        )
        self.assertEqual(
            self.con.execute("SELECT review_status FROM kg_candidate_relations WHERE id = ?", (candidate["id"],)).fetchone()[0],
            "accepted",
        )
        self.assertEqual(
            self.con.execute("SELECT COUNT(*) FROM kg_relations WHERE predicate = 'related_to'").fetchone()[0],
            1,
        )

        payload = semantic_graph_payload(self.con)
        self.assertFalse(any(node["type"] == "corpus" for node in payload["nodes"]))
        self.assertFalse(any(edge["predicate"] == "co_occurs_with" for edge in payload["edges"]))

    def test_registered_rules_match_edge_and_candidate_policies(self):
        build_semantic_graph(self.con)

        required_rules = {
            "structural.collection.metadata",
            "structural.source_type.metadata",
            "bibliographic.author.metadata",
            "semantic.topic.manual_keyword",
            "semantic.observation_topic.human",
            "spatial.place.metadata",
            "temporal.time.metadata",
            "candidate.cooccurrence",
            "semantic.related_to.reviewed_cooccurrence",
            "reserved.near_in_evidence",
        }
        self.assertTrue(required_rules.issubset(RELATION_RULES))
        rules_payload = semantic_relation_rules_payload()
        self.assertEqual(rules_payload["count"], len(RELATION_RULES))
        self.assertFalse(RELATION_RULES["candidate.cooccurrence"]["visible_by_default"])
        self.assertTrue(RELATION_RULES["candidate.cooccurrence"]["candidate_only"])
        self.assertEqual(RELATION_RULES["candidate.cooccurrence"]["default_status"], "needs_review")
        self.assertFalse(RELATION_RULES["reserved.near_in_evidence"]["implemented"])
        self.assertTrue(RELATION_RULES["reserved.near_in_evidence"]["candidate_only"])
        for relation in self.con.execute("SELECT * FROM kg_relations").fetchall():
            self.assertIn(
                primary_relation_rule_id(
                    relation["predicate"],
                    relation["extraction_method"],
                    {},
                ),
                RELATION_RULES,
            )
        for candidate in self.con.execute("SELECT * FROM kg_candidate_relations").fetchall():
            self.assertIn(
                candidate_rule_id(candidate["predicate"], candidate["candidate_method"]),
                RELATION_RULES,
            )

        expected_metadata_relations = {
            "belongs_to_collection": "structural.collection.metadata",
            "has_source_type": "structural.source_type.metadata",
            "authored_by": "bibliographic.author.metadata",
        }
        for predicate, rule_id in expected_metadata_relations.items():
            relation = self.con.execute(
                "SELECT * FROM kg_relations WHERE predicate = ? LIMIT 1",
                (predicate,),
            ).fetchone()
            self.assertIsNotNone(relation)
            self.assertEqual(relation["assertion_status"], "accepted")
            self.assertEqual(relation["extraction_method"], "metadata")
            self.assertIn(rule_id, relation["properties_json"])

        topic = self.con.execute(
            """
            SELECT *
            FROM kg_relations
            WHERE predicate = 'has_topic'
            ORDER BY evidence_count DESC
            LIMIT 1
            """
        ).fetchone()
        self.assertIsNotNone(topic)
        topic_evidence_count = self.con.execute(
            "SELECT COUNT(*) FROM kg_relation_evidence WHERE relation_id = ?",
            (topic["id"],),
        ).fetchone()[0]
        self.assertEqual(topic_evidence_count, topic["evidence_count"])
        self.assertGreater(topic_evidence_count, 0)
        explanation = explain_semantic_relation(self.con, topic["id"])
        self.assertTrue(explanation["rule_id"])
        self.assertTrue(explanation["methodological_caution"])
        self.assertTrue(explanation["what_it_does_not_mean"])
        self.assertEqual(len(explanation["evidence_items"]), topic_evidence_count)

        auto_topic = self.con.execute(
            """
            SELECT r.*
            FROM kg_relations r
            JOIN kg_entities concept ON concept.id = r.object_entity_id
            WHERE r.predicate = 'has_topic'
              AND concept.normalized_label = 'machine suggestion'
            """
        ).fetchone()
        self.assertIsNotNone(auto_topic)
        self.assertEqual(auto_topic["extraction_method"], "auto_keyword")
        self.assertEqual(auto_topic["assertion_status"], "needs_review")
        self.assertIn("semantic.topic.auto_keyword", auto_topic["properties_json"])

        pattern_time = self.con.execute(
            """
            SELECT r.*
            FROM kg_relations r
            JOIN kg_entities time_entity ON time_entity.id = r.object_entity_id
            WHERE r.predicate = 'mentions_time'
              AND time_entity.normalized_label = '2019'
            """
        ).fetchone()
        self.assertIsNotNone(pattern_time)
        self.assertEqual(pattern_time["extraction_method"], "pattern_extraction")
        self.assertEqual(pattern_time["assertion_status"], "needs_review")
        self.assertIn("temporal.time.pattern", pattern_time["properties_json"])

        candidate = self.con.execute(
            "SELECT * FROM kg_candidate_relations WHERE predicate = 'co_occurs_with' LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["review_status"], "needs_review")
        self.assertIn("candidate.cooccurrence", candidate["properties_json"])
        self.assertEqual(
            self.con.execute(
                "SELECT COUNT(*) FROM kg_relations WHERE predicate = 'co_occurs_with'"
            ).fetchone()[0],
            0,
        )
        self.assertEqual(
            self.con.execute(
                "SELECT COUNT(*) FROM kg_relations WHERE predicate = 'near_in_evidence'"
            ).fetchone()[0],
            0,
        )

    def test_unresolved_places_and_invalid_page_ranges_stay_outside_relations(self):
        build_semantic_graph(self.con)

        unresolved = self.con.execute(
            """
            SELECT *
            FROM kg_place_resolution
            WHERE normalized_label = 'mystery island'
            """
        ).fetchone()
        self.assertIsNotNone(unresolved)
        self.assertEqual(unresolved["resolution_status"], "unresolved")
        self.assertIsNone(unresolved["resolved_place_entity_id"])
        self.assertEqual(
            self.con.execute(
                """
                SELECT COUNT(*)
                FROM kg_relations r
                JOIN kg_entities target ON target.id = r.object_entity_id
                WHERE r.predicate = 'mentions_place'
                  AND target.normalized_label = 'mystery island'
                """
            ).fetchone()[0],
            0,
        )

        invalid = self.con.execute(
            "SELECT * FROM kg_time_resolution WHERE normalized_label = '100-101'"
        ).fetchone()
        self.assertIsNotNone(invalid)
        self.assertEqual(invalid["resolution_status"], "invalid")
        self.assertEqual(invalid["time_type"], "invalid_page_range")
        self.assertIsNone(invalid["resolved_time_entity_id"])
        self.assertEqual(
            self.con.execute(
                """
                SELECT COUNT(*)
                FROM kg_relations r
                JOIN kg_entities target ON target.id = r.object_entity_id
                WHERE r.predicate = 'mentions_time'
                  AND target.normalized_label = '100-101'
                """
            ).fetchone()[0],
            0,
        )

    def test_rules_and_relation_explanation_endpoints(self):
        build_semantic_graph(self.con)
        relation = self.con.execute(
            "SELECT * FROM kg_relations WHERE predicate = 'has_topic' ORDER BY evidence_count DESC LIMIT 1"
        ).fetchone()
        original_init = repository_api.init_repository_db
        original_connection = repository_api.get_connection
        repository_api.init_repository_db = lambda: None
        repository_api.get_connection = lambda: nullcontext(self.con)
        try:
            app = FastAPI()
            app.include_router(repository_api.router, prefix="/api/v1")
            client = TestClient(app)

            rules_response = client.get("/api/v1/repository/graph/semantic/rules")
            self.assertEqual(rules_response.status_code, 200)
            rules_body = rules_response.json()
            self.assertEqual(rules_body["count"], len(RELATION_RULES))
            self.assertTrue(
                any(rule["rule_id"] == "candidate.cooccurrence" for rule in rules_body["rules"])
            )

            explain_response = client.get(
                f"/api/v1/repository/graph/semantic/relations/{relation['id']}/explain"
            )
            self.assertEqual(explain_response.status_code, 200)
            explain_body = explain_response.json()
            self.assertEqual(explain_body["relation_id"], relation["id"])
            self.assertTrue(explain_body["rule_id"])
            self.assertTrue(explain_body["methodological_caution"])
            self.assertGreater(len(explain_body["evidence_items"]), 0)
        finally:
            repository_api.init_repository_db = original_init
            repository_api.get_connection = original_connection


if __name__ == "__main__":
    unittest.main()
