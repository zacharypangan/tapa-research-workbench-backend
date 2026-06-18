import sqlite3
import unittest

from app.repository.semantic_graph import (
    build_semantic_graph,
    review_semantic_candidate,
    review_semantic_relation,
    semantic_graph_payload,
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
        self.con = sqlite3.connect(":memory:")
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")
        create_source_schema(self.con)
        for index in range(10):
            material_id = f"material-{index + 1}"
            keywords = "barkcloth; weaving" if index < 2 else None
            self.con.execute(
                """
                INSERT INTO materials (
                    id, title, authors, year, source_type, collection,
                    abstract_or_notes, language, region, status,
                    created_at, updated_at, keywords
                )
                VALUES (?, ?, ?, '2020', 'publication', ?, ?, 'English', ?, 'metadata_complete', ?, ?, ?)
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
                ),
            )
        segment_rows = [
            ("material-1", "p. 1", "Barkcloth and weaving were recorded in 2019."),
            ("material-1", "pp. 100-101", "Barkcloth and weaving appear on pages 100-101."),
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


if __name__ == "__main__":
    unittest.main()
