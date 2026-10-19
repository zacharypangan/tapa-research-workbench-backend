"""Deterministic inference proposals remain bounded, traceable and non-factual."""
import copy
import unittest
from unittest.mock import patch

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


def literal_material_graph(*, polarity="positive"):
    graph = document_graph()
    add_entity(graph, "a", "Registered object", kind="Artifact", evidence_id="shared")
    add_entity(graph, "b", "wood", kind="Material", evidence_id="shared")
    graph["evidence"][0]["text"] = "The registered object's material is wood."
    graph["assertions"] = [{"assertion_id": "literal-claim", "source_id": "s1",
        "subject_id": "a", "predicate": "has_material", "value": "wood",
        "polarity": polarity, "epistemic_kind": "source_report", "scope": {"role": "reported construction material"},
        "evidence_ids": ["shared"]}]
    return graph


class GraphInferencesTest(unittest.TestCase):
    def test_exact_source_literal_is_reexpressed_without_mutating_original(self):
        graph = literal_material_graph()
        before = copy.deepcopy(graph)
        result = infer_connections(graph)
        self.assertEqual(graph, before)
        self.assertEqual(len(result["supported_relationships"]), 1)
        row = result["supported_relationships"][0]
        self.assertEqual((row["subject_id"], row["predicate"], row["object_id"]), ("a", "has_material", "b"))
        self.assertEqual(row["derived_from_assertion_id"], "literal-claim")
        self.assertEqual(row["premise_ids"], ["literal-claim"])
        self.assertEqual(row["polarity"], "positive")
        self.assertEqual(row["scope"], graph["assertions"][0]["scope"])
        self.assertEqual(row["origin"], "literal_relation_recovery")
        self.assertEqual(row["review_status"], "needs_review")
        self.assertFalse(row["fact_transfer"])
        self.assertFalse(pair(result["inferences"]))
        self.assertEqual(result, infer_connections(graph))

    def test_recovery_retains_negative_uncertain_and_conflicting_original_claims(self):
        for polarity in ("negative", "uncertain"):
            with self.subTest(polarity=polarity):
                graph = literal_material_graph(polarity=polarity)
                rows = infer_connections(graph)["supported_relationships"]
                self.assertEqual([row["polarity"] for row in rows], [polarity])
        graph = literal_material_graph()
        negative = copy.deepcopy(graph["assertions"][0])
        negative["assertion_id"], negative["polarity"] = "negative-claim", "negative"
        graph["assertions"].append(negative)
        rows = infer_connections(graph)["supported_relationships"]
        self.assertEqual({row["polarity"] for row in rows}, {"positive", "negative"})

    def test_recovery_scan_budget_never_resolves_a_partially_scanned_ambiguity(self):
        graph = literal_material_graph()
        add_entity(graph, "c", "wood", kind="Material", evidence_id="shared")
        with patch("app.repository.graph_inferences.MAX_RECOVERY_COMPARISONS", 1):
            result = infer_connections(graph)
        self.assertFalse(result["supported_relationships"])
        self.assertTrue(result["report"]["supported_relationships_truncated"])
        self.assertEqual(result["report"]["recovery_considered_candidates"], 1)

    def test_recovery_never_uses_generated_english_or_unquoted_literal_alone(self):
        graph = literal_material_graph()
        graph["entities"][1].update(label_original="木", label_en="wood")
        self.assertFalse(infer_connections(graph)["supported_relationships"])
        graph = literal_material_graph()
        graph["evidence"][0]["text"] = "The registered object was catalogued."
        self.assertFalse(infer_connections(graph)["supported_relationships"])
        graph["evidence"][0]["text"] = "The material is hardwood."
        self.assertFalse(infer_connections(graph)["supported_relationships"])

    def test_recovery_rejects_prose_wrong_types_and_non_source_hypotheses(self):
        for field, value in (("value", "wood from an unknown tree"),
                             ("predicate", "unregistered_material_predicate"),
                             ("epistemic_kind", "project_hypothesis"),
                             ("review_status", "rejected")):
            with self.subTest(field=field):
                graph = literal_material_graph()
                graph["assertions"][0][field] = value
                self.assertFalse(infer_connections(graph)["supported_relationships"])
        graph = literal_material_graph()
        graph["entities"][1]["entity_type"] = "Person"
        self.assertFalse(infer_connections(graph)["supported_relationships"])

    def test_recovery_requires_unique_same_source_same_passage_target(self):
        graph = literal_material_graph()
        add_entity(graph, "c", "wood", kind="Material", evidence_id="shared")
        result = infer_connections(graph)
        self.assertFalse(result["supported_relationships"])
        self.assertEqual(result["report"]["ambiguous_literal_targets"], 1)
        graph = literal_material_graph()
        target_quote = copy.deepcopy(graph["evidence"][0])
        target_quote.update(evidence_id="other-quote", unit_id="other-unit")
        graph["evidence"].append(target_quote)
        graph["entities"][1]["evidence_ids"] = ["other-quote"]
        self.assertFalse(infer_connections(graph)["supported_relationships"])
        target_quote.update(source_id="s2")
        graph["entities"][1]["source_id"] = "s2"
        self.assertFalse(infer_connections(graph)["supported_relationships"])

    def test_recovery_respects_entity_rejections_and_does_not_duplicate_direct_edge(self):
        graph = literal_material_graph()
        graph["decisions"] = [{"candidate_id": "b", "decision": "reject"}]
        self.assertFalse(infer_connections(graph)["supported_relationships"])
        graph = literal_material_graph()
        direct = copy.deepcopy(graph["assertions"][0])
        direct.update(assertion_id="direct", object_id="b")
        direct.pop("value")
        graph["assertions"].append(direct)
        self.assertFalse(infer_connections(graph)["supported_relationships"])

    def test_physical_objects_people_and_events_do_not_align_by_name(self):
        for kind in ("Artifact", "ArchaeologicalObject", "ToolInstance", "MaterialBatch", "Person", "Activity", "ReportedStage", "ProductionCase", "Document"):
            with self.subTest(kind=kind):
                graph = document_graph()
                add_entity(graph, "a", "Shared recorded name", kind=kind)
                add_entity(graph, "b", "Shared recorded name", kind=kind, source="s2")
                result = infer_connections(graph)
                self.assertFalse(pair(result["inferences"]))
                self.assertEqual(result["report"]["nonalignable_entities"], 2)

    def test_place_and_time_name_correspondence_does_not_claim_identity(self):
        for kind in ("Place", "TimeExpression"):
            with self.subTest(kind=kind):
                graph = document_graph()
                add_entity(graph, "a", "Shared source term", kind=kind)
                add_entity(graph, "b", "Shared source term", kind=kind, source="s2")
                row = pair(infer_connections(graph)["inferences"])[0]
                self.assertEqual(row["predicate"], "has_related_concept_context")
                self.assertEqual(row["mapping_kind"], "reported_name_correspondence")
                self.assertEqual(row["layer"], "concept_alignment")

    def test_provenance_retains_seed_family_and_origin_without_independence_claim(self):
        graph = document_graph()
        graph["sources"][0].update(origin="primary", source_family_id="museum-catalogue", root_seed_ids=["paper-root"])
        graph["sources"][1].update(origin="web", source_family_id="museum-catalogue", seed_ids=["paper-question"])
        add_entity(graph, "a", "Paper mulberry", kind="Material")
        add_entity(graph, "b", "Paper mulberry", kind="Material", source="s2")
        row = pair(infer_connections(graph)["inferences"])[0]
        self.assertEqual(row["source_ids"], ["s1", "s2"])
        self.assertEqual(row["source_family_ids"], ["museum-catalogue"])
        self.assertEqual(row["source_independence"], "same_declared_family")
        self.assertEqual(row["root_seed_ids"], ["paper-root"])
        self.assertEqual(row["seed_ids"], ["paper-question"])
        self.assertEqual(row["primary_entity_ids"], ["a"])
        self.assertTrue(row["connects_primary_to_augmented"])
        graph["sources"][1]["source_family_id"] = "another-catalogue"
        row = pair(infer_connections(graph)["inferences"])[0]
        self.assertEqual(row["source_independence"], "not_established")
        graph["sources"][1]["origin"] = "curation"
        row = pair(infer_connections(graph)["inferences"])[0]
        self.assertFalse(row["connects_primary_to_augmented"])
        graph["sources"][1]["source_kind"] = "web_snapshot"
        row = pair(infer_connections(graph)["inferences"])[0]
        self.assertTrue(row["connects_primary_to_augmented"])

    def test_exact_source_definition_can_bridge_different_reported_names(self):
        graph = document_graph()
        add_entity(graph, "a", "masi", kind="Concept")
        add_entity(graph, "b", "barkcloth", kind="Material", source="s2")
        graph["evidence"][0]["text"] = "We use the term masi for barkcloth."
        graph["assertions"] = [{"assertion_id": "definition", "source_id": "s1", "subject_id": "a",
            "predicate": "has_definition", "value": "barkcloth", "polarity": "positive",
            "epistemic_kind": "source_report", "scope": {}, "evidence_ids": ["quote-a"]}]
        row = pair(infer_connections(graph)["inferences"])[0]
        self.assertEqual(row["rule"], "source_definition_name_match")
        self.assertEqual(row["premise_ids"], ["definition"])
        self.assertEqual(row["predicate"], "has_related_concept_context")
        self.assertEqual(row["polarity"], "uncertain")
        for field, value in (("polarity", "negative"), ("review_status", "rejected"), ("value", "a kind of barkcloth")):
            with self.subTest(field=field):
                changed = copy.deepcopy(graph)
                changed["assertions"][0][field] = value
                self.assertFalse(pair(infer_connections(changed)["inferences"]))
        for entity_index in (0, 1):
            changed = copy.deepcopy(graph)
            changed["entities"][entity_index]["is_instance"] = True
            self.assertFalse(pair(infer_connections(changed)["inferences"]))

    def test_typology_codes_and_taxonomic_ranks_remain_separate(self):
        graph = document_graph()
        add_entity(graph, "a", "Type I", kind="TypeCategory")
        add_entity(graph, "b", "Type I", kind="TypeCategory", source="s2")
        self.assertFalse(pair(infer_connections(graph)["inferences"]))
        for entity in graph["entities"]:
            entity["scope"]["scheme_id"] = "same-reported-scheme"
        self.assertTrue(pair(infer_connections(graph)["inferences"]))
        graph = document_graph()
        add_entity(graph, "a", "Mulberry", kind="PlantTaxon", scope={"rank": "species"})
        add_entity(graph, "b", "Mulberry", kind="PlantTaxon", source="s2", scope={"rank": "genus"})
        self.assertFalse(pair(infer_connections(graph)["inferences"]))

    def test_numeric_only_labels_do_not_create_concept_correspondence(self):
        graph = document_graph()
        add_entity(graph, "a", "123", kind="Concept")
        add_entity(graph, "b", "123", kind="Concept", source="s2")
        self.assertFalse(pair(infer_connections(graph)["inferences"]))

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
                       {"max_connections": True}, {"max_per_entity": 1.5},
                       {"max_supported_relationships": 0}, {"max_supported_relationships": True}):
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
        graph = shared_target_graph()
        graph["assertions"][0]["source_id"] = "s2"
        with self.assertRaisesRegex(ValueError, "another source"):
            infer_connections(graph)


if __name__ == "__main__":
    unittest.main()
