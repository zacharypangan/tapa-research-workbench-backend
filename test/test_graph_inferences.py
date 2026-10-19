"""Deterministic inference proposals remain bounded, traceable and non-factual."""
import copy
import unittest

from app.repository.graph_inferences import infer_connections


def document_graph():
    return {"sources": [{"source_id": "s1"}, {"source_id": "s2"}],
            "entities": [], "evidence": [], "assertions": [], "alignments": [],
            "decisions": []}


def add_entity(graph, key, label, *, source="s1", kind="Place", unit=None,
               start=0, scope=None, english=None, evidence_id=None):
    evidence_id = evidence_id or "quote-" + key
    if not any(row["evidence_id"] == evidence_id for row in graph["evidence"]):
        text = "An original source quotation."
        graph["evidence"].append({"evidence_id": evidence_id, "source_id": source,
            "unit_id": unit or "unit-" + key, "text": text,
            "character_locator": {"coordinate_system": "unit_text", "offset_unit": "unicode_code_points",
                                  "spans": [{"start": start, "end": start + len(text)}]},
            "locator": {"page": 1}})
    entity = {"entity_id": key, "source_id": source, "entity_type": kind,
              "label_original": label, "evidence_ids": [evidence_id], "scope": scope or {}}
    if english:
        entity["label_en"] = english
    graph["entities"].append(entity)
    return entity


def pair(rows, left="a", right="b"):
    return [row for row in rows if {row["subject_id"], row["object_id"]} == {left, right}]


def shared_target_graph():
    graph = document_graph()
    add_entity(graph, "a", "First object", kind="Artifact")
    add_entity(graph, "b", "Second object", kind="Artifact")
    add_entity(graph, "c", "Reported location")
    graph["assertions"] = [{"assertion_id": "claim-" + key, "source_id": "s1",
        "subject_id": key, "object_id": "c", "predicate": "located_in", "polarity": "positive",
        "evidence_ids": ["quote-" + key, "quote-c"], "scope": {}}
        for key in ("a", "b")]
    return graph


class GraphInferencesTest(unittest.TestCase):
    def test_matching_labels_are_uncertain_proposals_with_own_evidence(self):
        graph = document_graph()
        add_entity(graph, "a", "石 拍", kind="ArtifactType")
        add_entity(graph, "b", "石拍", source="s2", kind="ArtifactType")
        rows = pair(infer_connections(graph)["inferences"])
        self.assertEqual(len(rows), 1)
        proposal = rows[0]
        self.assertEqual(proposal["polarity"], "uncertain")
        self.assertEqual(proposal["epistemic_kind"], "inference")
        self.assertEqual(proposal["review_status"], "needs_review")
        self.assertFalse(proposal["fact_transfer"])
        self.assertEqual(set(proposal["evidence_ids"]), {"quote-a", "quote-b"})
        self.assertTrue(proposal["rule"])
        self.assertTrue(proposal["rationale"])
        self.assertEqual(len(graph["entities"]), 2)

    def test_english_alias_and_nfkc_case_normalization(self):
        graph = document_graph()
        add_entity(graph, "a", "台湾", english="Ｔａｉｗａｎ")
        add_entity(graph, "b", "TAIWAN", source="s2")
        self.assertTrue(pair(infer_connections(graph)["inferences"]))

    def test_type_conflicts_and_generic_stopwords_do_not_link(self):
        for first_kind, second_kind, label in (("Place", "Person", "Jordan"),
                                                ("Concept", "Concept", "the")):
            with self.subTest(label=label):
                graph = document_graph()
                add_entity(graph, "a", label, kind=first_kind)
                add_entity(graph, "b", label, source="s2", kind=second_kind)
                self.assertFalse(pair(infer_connections(graph)["inferences"]))

    def test_scope_conflicts_block_same_label_hypotheses(self):
        for key in ("place_id", "time_id", "scheme_id", "language", "namespace"):
            with self.subTest(key=key):
                graph = document_graph()
                add_entity(graph, "context-a", "First context")
                add_entity(graph, "context-b", "Second context")
                add_entity(graph, "a", "Shared term", scope={key: "context-a"})
                add_entity(graph, "b", "Shared term", source="s2", scope={key: "context-b"})
                self.assertFalse(pair(infer_connections(graph)["inferences"]))

    def test_exact_shared_quote_and_nearby_offsets_produce_context_only(self):
        for same_quote in (True, False):
            with self.subTest(same_quote=same_quote):
                graph = document_graph()
                add_entity(graph, "a", "Stone beater", kind="Artifact", unit="same-unit")
                add_entity(graph, "b", "Paper mulberry", kind="Material", unit="same-unit", start=200,
                           evidence_id="quote-a" if same_quote else "quote-b")
                rows = pair(infer_connections(graph)["inferences"])
                self.assertTrue(rows)
                self.assertTrue(all(row["polarity"] == "uncertain" and row["fact_transfer"] is False
                                    and row["premise_ids"] == [] for row in rows))

    def test_context_never_crosses_sources_or_distant_unit_offsets(self):
        for source, start in (("s2", 50), ("s1", 900)):
            with self.subTest(source=source, start=start):
                graph = document_graph()
                add_entity(graph, "a", "Stone beater", kind="Artifact", unit="same-unit")
                add_entity(graph, "b", "Paper mulberry", kind="Material", unit="same-unit", start=start,
                           source=source)
                self.assertFalse(pair(infer_connections(graph)["inferences"]))

    def test_multiple_quotations_keep_nearby_match_independent_of_reference_order(self):
        graph = document_graph()
        first = add_entity(graph, "a", "Stone beater", kind="Artifact", unit="same-unit")
        add_entity(graph, "b", "Paper mulberry", kind="Material", unit="same-unit", start=200)
        distant = copy.deepcopy(graph["evidence"][0])
        distant["evidence_id"] = "quote-a-distant"
        distant["character_locator"]["spans"] = [{"start": 2000, "end": 2028}]
        graph["evidence"].append(distant)
        first["evidence_ids"].append(distant["evidence_id"])
        result = infer_connections(graph)
        self.assertTrue(pair(result["inferences"]))
        first["evidence_ids"].reverse()
        self.assertEqual(result, infer_connections(graph))

    def test_positive_shared_target_retains_both_premises(self):
        graph = shared_target_graph()
        rows = pair(infer_connections(graph)["inferences"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]["premise_ids"]), {"claim-a", "claim-b"})
        self.assertEqual(set(rows[0]["evidence_ids"]), {"quote-a", "quote-b", "quote-c"})
        self.assertFalse(rows[0]["fact_transfer"])

    def test_nonpositive_or_rejected_premises_do_not_propagate(self):
        for field, value in (("polarity", "negative"), ("polarity", "uncertain"),
                             ("review_status", "rejected"), ("review_status", "excluded")):
            with self.subTest(field=field, value=value):
                graph = shared_target_graph()
                graph["assertions"][0][field] = value
                self.assertFalse(pair(infer_connections(graph)["inferences"]))
        graph = shared_target_graph()
        graph["decisions"] = [{"candidate_id": "claim-a", "decision": "reject",
            "at": "2026-10-19T10:00:00Z", "rationale": "Wrong attribution"}]
        self.assertFalse(pair(infer_connections(graph)["inferences"]))

    def test_rejected_entity_is_not_in_an_inference(self):
        for direct in (False, True):
            with self.subTest(direct=direct):
                graph = shared_target_graph()
                if direct:
                    graph["entities"][0]["review_status"] = "excluded"
                else:
                    graph["decisions"] = [{"candidate_id": "a", "decision": "reject",
                        "at": "2026-10-19T10:00:00Z", "rationale": "Wrong entity"}]
                self.assertFalse(any("a" in (row["subject_id"], row["object_id"])
                                     for row in infer_connections(graph)["inferences"]))

    def test_latest_dated_review_controls_premise_eligibility(self):
        graph = shared_target_graph()
        earlier = {"candidate_id": "claim-a", "decision": "reject", "rationale": "Initial check",
                   "at": "2026-10-19T09:00:00Z"}
        later = {"candidate_id": "claim-a", "decision": "accept", "rationale": "Corrected check",
                 "at": "2026-10-19T10:00:00Z"}
        graph["decisions"] = [later, earlier]
        self.assertTrue(pair(infer_connections(graph)["inferences"]))
        earlier["decision"], later["decision"] = "accept", "reject"
        self.assertFalse(pair(infer_connections(graph)["inferences"]))

    def test_review_dates_compare_instants_across_timezones(self):
        graph = shared_target_graph()
        graph["decisions"] = [
            {"candidate_id": "claim-a", "decision": "reject", "rationale": "Earlier",
             "at": "2026-10-19T10:00:00+09:00"},
            {"candidate_id": "claim-a", "decision": "accept", "rationale": "Later",
             "at": "2026-10-19T02:00:00Z"}]
        self.assertTrue(pair(infer_connections(graph)["inferences"]))

    def test_exactly_filling_limit_without_more_proposals_is_not_truncated(self):
        graph = document_graph()
        add_entity(graph, "a", "Fiji")
        add_entity(graph, "b", "Fiji", source="s2")
        result = infer_connections(graph, max_connections=1, max_per_entity=1)
        self.assertEqual(len(result["inferences"]), 1)
        self.assertFalse(result["report"]["truncated"])

    def test_output_is_deterministic_bounded_and_does_not_mutate_input(self):
        graph = document_graph()
        for index in range(9):
            add_entity(graph, "n" + str(index), "Fiji")
        before = copy.deepcopy(graph)
        result = infer_connections(graph, max_connections=4, max_per_entity=2)
        self.assertEqual(graph, before)
        self.assertEqual(len(result["inferences"]), 4)
        self.assertTrue(result["report"]["truncated"])
        degree = {}
        ids = []
        for row in result["inferences"]:
            self.assertNotEqual(row["subject_id"], row["object_id"])
            ids.append(row["inference_id"])
            for key in ("subject_id", "object_id"):
                degree[row[key]] = degree.get(row[key], 0) + 1
        self.assertEqual(len(set(ids)), len(ids))
        self.assertLessEqual(max(degree.values()), 2)
        self.assertEqual(result, infer_connections(graph, max_connections=4, max_per_entity=2))
        for field in ("entities", "evidence", "sources"):
            graph[field].reverse()
        self.assertEqual(result, infer_connections(graph, max_connections=4, max_per_entity=2))

    def test_invalid_limits_and_unresolved_identifiers_fail_closed(self):
        for kwargs in ({"max_connections": 0}, {"max_connections": 5001},
                       {"max_per_entity": 0}, {"max_per_entity": 51},
                       {"max_connections": True}, {"max_per_entity": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                infer_connections(document_graph(), **kwargs)
        graph = shared_target_graph()
        graph["entities"].append(copy.deepcopy(graph["entities"][0]))
        with self.assertRaises(ValueError):
            infer_connections(graph)
        graph = shared_target_graph()
        graph["assertions"][0]["object_id"] = "missing"
        with self.assertRaises(ValueError):
            infer_connections(graph)
        graph = shared_target_graph()
        graph["entities"][0]["evidence_ids"] = ["missing"]
        with self.assertRaises(ValueError):
            infer_connections(graph)


if __name__ == "__main__":
    unittest.main()
