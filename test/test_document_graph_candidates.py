"""No provider/network access: stage responses are source-addressed fixtures."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

import httpx

from app.repository import document_graph_candidates as candidates


TEXT = "Case A reports uncertain beating with a stone beater, not wood. 石拍见于台湾。"
ONTOLOGY = {
    "version": "fixture-v1",
    "entity_types": {"ProductionCase": "A documented case", "ActivityType": "An activity",
                     "ToolForm": "A form", "MaterialKind": "A material", "Place": "A place",
                     "TimeExpression": "A substantive date"},
    "predicates": {
        "uses_tool_form": {"domain": ["ActivityType"], "range": ["ToolForm"],
                           "requires_object": True, "requires_case_scope": True},
        "uses_material": {"domain": ["ActivityType"], "range": ["MaterialKind"], "requires_object": True},
        "has_definition": {},
    },
}
LABELS = {"Case A": "ProductionCase", "beating": "ActivityType", "stone beater": "ToolForm", "石拍": "ToolForm"}


def unit(text=TEXT, key="p7"):
    return {"unit_id": key, "source_id": "PAPER", "source_version_id": "PAPER:v1",
            "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "locator": {"pdf_page": 7}, "seed_ids": ["root-seed"]}


class Fixture:
    def __init__(self, mutate=None, labels=None, status="supported"):
        self.requests = []
        self.mutate = mutate
        self.labels = labels or LABELS
        self.status = status

    def __call__(self, request):
        body = json.loads(request.content)
        prompt = json.loads(body["messages"][1]["content"])
        self.requests.append((request, body, prompt))
        stage, data = prompt["stage"], prompt["input"]
        source = data["source_spans"]
        if stage == "mentions":
            mentions = []
            for label in self.labels:
                for span in source:
                    if label not in span["text"]:
                        continue
                    start = span["text"].index(label)
                    end = start + len(label)
                    tokens = candidates._tokens({**span, "start_char": 0})
                    first = next(token["token_id"] for token in tokens if token["start_char"] == start)
                    last = next(token["token_id"] for token in tokens if token["end_char"] == end)
                    mentions.append({"span_id": span["span_id"], "start_token": first, "end_token": last})
            result = {"mentions": mentions, "complete": True}
        elif stage == "typing":
            result = {"types": [{"mention_id": item["mention_id"], "entity_type": self.labels[item["text"]],
                                 "scope": {}, "label_en": "stone beater" if item["text"] == "石拍" else None}
                                for item in data["mentions"]], "unresolved": [], "complete": True}
        elif stage in {"relations", "recall"}:
            by_label = {item["label_original"]: item["local_id"] for item in data["entities"]}
            result = {"relations": [], "complete": True}
            if stage == "recall":
                result["unresolved_questions"] = []
            elif {"beating", "stone beater", "Case A"} <= set(by_label):
                result["relations"] = [{"subject": by_label["beating"], "predicate": "uses_tool_form",
                    "object": by_label["stone beater"], "scope": {"case_id": by_label["Case A"]},
                    "polarity": "uncertain", "epistemic_kind": "source_report",
                    "evidence_span_ids": [source[0]["span_id"]],
                    "rationale": "The original source explicitly reports uncertainty."}]
        else:
            result = {"reviews": [{"claim_id": item["local_id"], "status": self.status,
                                    "reason": "Check original source and case attribution."}
                                   for item in data["claims"]], "complete": True}
        if self.mutate:
            changed = self.mutate(stage, result, prompt, body)
            if changed is not None:
                result = changed
        if isinstance(result, httpx.Response):
            return result
        content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return httpx.Response(200, json={"model": "test-model", "done": True, "done_reason": "stop",
            "message": {"content": content},
            "workbench_model_audit": {"requested_model": "test-model", "policy_version": "fixture-policy"}})


class DocumentGraphCandidatesTests(unittest.IsolatedAsyncioTestCase):
    async def generate(self, fixture=None, units=None, **kwargs):
        fixture = fixture or Fixture()
        result = await candidates.provider(units or [unit()], copy.deepcopy(ONTOLOGY),
            kwargs.get("base_url", "https://gateway.invalid/model-gateway"), "test-model",
            {"Authorization": "Bearer test-only-secret"}, transport=httpx.MockTransport(fixture))
        return result, fixture

    async def test_stages_preserve_original_anchors_polarity_and_review_status(self):
        inputs, ontology = [unit()], copy.deepcopy(ONTOLOGY)
        original = copy.deepcopy(inputs)
        result, fixture = await self.generate(units=inputs)
        self.assertEqual([entry[2]["stage"] for entry in fixture.requests],
                         ["mentions", "typing", "relations", "recall", "verify"])
        self.assertEqual(inputs, original)
        self.assertEqual(ONTOLOGY, ontology)
        self.assertEqual(len(result["entities"]), 4)
        self.assertEqual(len(result["assertions"]), 1)
        claim = result["assertions"][0]
        self.assertEqual(claim["polarity"], "uncertain")
        self.assertEqual(claim["review_status"], "needs_specialist_review")
        self.assertEqual(claim["scientific_status"], "provisional_extraction")
        self.assertEqual(result["rejections"], [])
        self.assertTrue(result["provider_metadata"]["complete"])
        self.assertTrue(result["provider_metadata"]["semantic_verification_is_diagnostic"])
        self.assertEqual(result["provider_metadata"]["model_digest_status"], "not_supplied_by_provider")
        self.assertEqual(result["provider_metadata"]["gateway_audits"][0]["policy_version"], "fixture-policy")
        chinese = next(item for item in result["entities"] if item["label_original"] == "石拍")
        self.assertEqual(chinese["label_en_status"], "model_translation_pending_review")
        for record in result["evidence"]:
            for span in record["character_locator"]["spans"]:
                self.assertEqual(TEXT[span["start"]:span["end"]], record["text"])
        for request, body, prompt in fixture.requests:
            self.assertEqual(str(request.url), "https://gateway.invalid/model-gateway/api/chat")
            self.assertEqual(body["options"]["num_ctx"], 16384)
            self.assertLessEqual(body["options"]["num_predict"], 4096)
            self.assertEqual(body["format"], prompt["output_schema"])
        self.assertNotIn("test-only-secret", json.dumps(result))

    async def test_duplicate_json_keys_fail_visibly_without_retry(self):
        fixture = Fixture(lambda stage, *_: '{"mentions":[],"complete":true,"complete":false}' if stage == "mentions" else None)
        result, fixture = await self.generate(fixture)
        self.assertEqual(len(fixture.requests), 1)
        self.assertFalse(result["provider_metadata"]["complete"])
        self.assertIn("duplicate_json_key", result["rejections"][0]["reason"])

    async def test_bad_mention_preserves_good_siblings(self):
        def mutate(stage, answer, *_):
            if stage == "mentions":
                answer["mentions"].append({"span_id": "S1", "start_token": 0, "end_token": 9999})
        result, _ = await self.generate(Fixture(mutate))
        self.assertEqual(len(result["entities"]), 4)
        self.assertEqual(len(result["assertions"]), 1)
        self.assertTrue(any("token_range_out_of_bounds" in item["reason"] for item in result["rejections"]))
        self.assertFalse(result["provider_metadata"]["complete"])

    async def test_bare_date_rejected_without_losing_other_typed_mentions(self):
        labels = {**LABELS, "circa": "TimeExpression"}
        result, _ = await self.generate(Fixture(labels=labels), units=[unit(TEXT + " circa")])
        self.assertEqual(len(result["entities"]), 4)
        self.assertEqual(len(result["assertions"]), 1)
        self.assertTrue(any("bare_approximation" in item["reason"] for item in result["rejections"]))

    async def test_invented_scope_preserves_other_entities(self):
        def mutate(stage, answer, *_):
            if stage == "typing":
                answer["types"][-1]["scope"] = {"place_id": "invented-place-id"}
        result, _ = await self.generate(Fixture(mutate))
        self.assertEqual(len(result["entities"]), 3)
        self.assertEqual(len(result["assertions"]), 1)
        self.assertTrue(any("scope_id_not_registered" in item["reason"] for item in result["rejections"]))

    async def test_unknown_scope_fields_are_rejected_instead_of_stripped(self):
        def mutate(stage, answer, *_):
            if stage == "typing":
                answer["types"][-1]["scope"] = {"text": "石拍"}
        result, _ = await self.generate(Fixture(mutate))
        self.assertEqual(len(result["entities"]), 3)
        self.assertTrue(any("scope_fields_not_supported" in item["reason"] for item in result["rejections"]))

    async def test_full_ontology_type_constraints_remain_authoritative(self):
        def mutate(stage, answer, *_):
            if stage == "relations":
                answer["relations"][0]["predicate"] = "uses_material"
        result, _ = await self.generate(Fixture(mutate))
        self.assertEqual(len(result["assertions"]), 0)
        self.assertTrue(any("object type violates" in item["reason"] for item in result["rejections"]))

    async def test_relation_evidence_must_contain_both_endpoints(self):
        def mutate(stage, answer, prompt, *_):
            if stage == "relations":
                answer["relations"][0]["evidence_span_ids"] = [prompt["input"]["source_spans"][-1]["span_id"]]
        result, _ = await self.generate(Fixture(mutate), units=[unit(), unit("An unrelated passage.", "p8")])
        self.assertEqual(result["assertions"], [])
        self.assertTrue(any("quote_missing_endpoint" in item["reason"] for item in result["rejections"]))

    async def test_bad_relation_does_not_discard_good_sibling(self):
        def mutate(stage, answer, *_):
            if stage == "relations":
                bad = copy.deepcopy(answer["relations"][0])
                bad["object"] = "invented-object"
                answer["relations"].append(bad)
        result, _ = await self.generate(Fixture(mutate))
        self.assertEqual(len(result["assertions"]), 1)
        self.assertTrue(any("object_not_registered" in item["reason"] for item in result["rejections"]))

    async def test_semantic_unsupported_withheld_but_raw_response_retained(self):
        result, _ = await self.generate(Fixture(status="unsupported"))
        self.assertEqual(result["assertions"], [])
        self.assertEqual(result["provider_metadata"]["semantic_diagnostics"][0]["status"], "unsupported")
        self.assertTrue(result["provider_attempts"][2]["stage_response"]["relations"])
        self.assertFalse(result["provider_metadata"]["complete"])

    async def test_ambiguous_verification_stays_provisional(self):
        result, _ = await self.generate(Fixture(status="needs_review"))
        self.assertEqual(len(result["assertions"]), 1)
        self.assertEqual(result["assertions"][0]["review_status"], "needs_specialist_review")

    async def test_verification_omission_never_imports_unreviewed_claim(self):
        def mutate(stage, answer, *_):
            if stage == "verify":
                answer["reviews"] = []
        result, _ = await self.generate(Fixture(mutate))
        self.assertEqual(result["assertions"], [])
        self.assertTrue(any("verification_coverage_incomplete" in item["reason"] for item in result["rejections"]))

    async def test_new_referents_from_recall_remain_explicitly_unresolved(self):
        def mutate(stage, answer, *_):
            if stage == "recall":
                answer["unresolved_questions"] = ["Does wood need its own registered material mention?"]
        result, _ = await self.generate(Fixture(mutate))
        self.assertEqual(len(result["entities"]), 4)
        self.assertTrue(result["provider_metadata"]["coverage"][0]["unresolved_questions"])
        self.assertFalse(result["provider_metadata"]["complete"])

    async def test_recall_http_failure_does_not_discard_verified_initial_claims(self):
        def mutate(stage, *_):
            return httpx.Response(400, text="fixture recall failure") if stage == "recall" else None
        result, fixture = await self.generate(Fixture(mutate))
        self.assertEqual([entry[2]["stage"] for entry in fixture.requests],
                         ["mentions", "typing", "relations", "recall", "verify"])
        self.assertEqual(len(result["assertions"]), 1)
        self.assertTrue(any("staged_recall:provider_http_400" in item["reason"] for item in result["rejections"]))
        self.assertFalse(result["provider_metadata"]["complete"])

    async def test_recall_prompt_budget_failure_still_verifies_initial_claims(self):
        original = candidates._stage_schema
        def inflate_recall(schema, stage, data):
            result = original(schema, stage, data)
            if stage == "recall":
                result["description"] = "x" * candidates.LIMITS["prompt_bytes"]
            return result
        with patch.object(candidates, "_stage_schema", side_effect=inflate_recall):
            result, fixture = await self.generate()
        self.assertEqual([entry[2]["stage"] for entry in fixture.requests],
                         ["mentions", "typing", "relations", "verify"])
        self.assertEqual(len(result["assertions"]), 1)
        self.assertTrue(any("staged_recall:prompt_budget_exceeded" in item["reason"] for item in result["rejections"]))

    async def test_recall_and_verification_claim_inputs_are_compact(self):
        result, fixture = await self.generate()
        recall = next(item[2]["input"] for item in fixture.requests if item[2]["stage"] == "recall")
        verify = next(item[2]["input"] for item in fixture.requests if item[2]["stage"] == "verify")
        for claim in recall["existing_claims"] + verify["claims"]:
            self.assertNotIn("evidence", claim)
            self.assertNotIn("rationale", claim)
        self.assertEqual(verify["claims"][0]["evidence_span_ids"], ["S1"])
        self.assertIn("evidence", result["provider_response"]["assertions"][0])

    async def test_stage_schemas_use_only_current_server_selectors(self):
        _, fixture = await self.generate()
        for _, _, prompt in fixture.requests:
            stage, data, schema = prompt["stage"], prompt["input"], prompt["output_schema"]
            if stage == "mentions":
                fields = schema["properties"]["mentions"]["items"]["properties"]
                self.assertEqual(fields["span_id"]["enum"], ["S1"])
                self.assertEqual(set(fields), {"span_id", "start_token", "end_token"})
                self.assertIn("numbered_tokens", data["source_spans"][0])
            elif stage == "typing":
                scope = schema["properties"]["types"]["items"]["properties"]["scope"]
                self.assertFalse(scope["additionalProperties"])
                self.assertNotIn("text", scope["properties"])
            elif stage in {"relations", "recall"}:
                fields = schema["properties"]["relations"]["items"]["properties"]
                ids = [entity["local_id"] for entity in data["entities"]]
                self.assertEqual(fields["subject"]["enum"], ids)
                self.assertEqual(fields["object"]["enum"], ids)
                self.assertEqual(fields["evidence_span_ids"]["items"]["enum"], ["S1"])

    async def test_http_failure_no_external_fallback_or_retry(self):
        result, fixture = await self.generate(Fixture(lambda *_: httpx.Response(429, text="rate limited")))
        self.assertEqual(len(fixture.requests), 1)
        self.assertIn("provider_http_429", result["rejections"][0]["reason"])
        self.assertFalse(result["provider_metadata"]["complete"])

    async def test_output_truncation_never_looks_completed(self):
        def mutate(*_):
            return httpx.Response(200, json={"model": "test-model", "done": True, "done_reason": "length",
                                            "message": {"content": '{"mentions":[],"complete":true}'}})
        result, _ = await self.generate(Fixture(mutate))
        self.assertTrue(any("provider_output_incomplete" in item["reason"] for item in result["rejections"]))

    async def test_exhausted_call_budget_is_visible(self):
        with patch.dict(candidates.LIMITS, {"max_calls": 1}):
            result, fixture = await self.generate()
        self.assertEqual(len(fixture.requests), 1)
        self.assertTrue(any("call_budget_exhausted" in item["reason"] for item in result["rejections"]))

    async def test_saturated_mention_allowance_does_not_claim_complete(self):
        with patch.dict(candidates.LIMITS, {"mentions_per_window": 4}):
            result, _ = await self.generate()
        self.assertEqual(len(result["entities"]), 4)
        self.assertTrue(any("mention_allowance_reached" in item["reason"] for item in result["rejections"]))
        self.assertFalse(result["provider_metadata"]["complete"])

    async def test_saturated_relation_allowance_does_not_claim_complete(self):
        with patch.dict(candidates.LIMITS, {"relations_per_window": 1}):
            result, _ = await self.generate()
        self.assertEqual(len(result["assertions"]), 1)
        self.assertTrue(any("relation_allowance_reached" in item["reason"] for item in result["rejections"]))
        self.assertFalse(result["provider_metadata"]["complete"])

    async def test_all_unprocessed_source_windows_are_accounted_for(self):
        with patch.dict(candidates.LIMITS, {"window_bytes": 80, "max_windows": 1}):
            result, _ = await self.generate(units=[unit(TEXT * 40)])
        meta = result["provider_metadata"]
        self.assertGreater(meta["total_windows"], 1)
        self.assertEqual(len(meta["coverage"]), meta["total_windows"])
        self.assertTrue(all(row["status"] == "unprocessed" for row in meta["coverage"][1:]))
        accounted = {(span["unit_id"], span["start_char"], span["end_char"])
                     for row in meta["coverage"] for span in row["original_ranges"]}
        expected = {(span["unit_id"], span["start_char"], span["end_char"])
                    for span in candidates._windows([unit(TEXT * 40)])[0]}
        self.assertEqual(accounted, expected)
        self.assertFalse(meta["complete"])

    async def test_no_mentions_is_not_semantic_absence_proof(self):
        def mutate(stage, *_):
            return {"mentions": [], "complete": True} if stage == "mentions" else None
        result, fixture = await self.generate(Fixture(mutate))
        self.assertEqual(len(fixture.requests), 1)
        self.assertEqual(result["provider_metadata"]["coverage"][0]["semantic_completeness"], "unverified")
        self.assertEqual(result["entities"], [])

    async def test_cloud_requests_json_without_schema_but_keeps_schema_in_prompt(self):
        _, fixture = await self.generate(base_url="https://ollama.com")
        self.assertEqual(fixture.requests[0][1]["format"], "json")
        self.assertIn("output_schema", fixture.requests[0][2])

    async def test_gpt_oss_uses_low_thinking_and_4096_output_budget(self):
        captured = []
        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"model": "gpt-oss:120b", "done": True,
                "message": {"content": '{"mentions":[],"complete":true}'}})
        result = await candidates.provider([unit()], ONTOLOGY, "https://ollama.com", "gpt-oss:120b",
                                           transport=httpx.MockTransport(handler))
        self.assertEqual(captured[0]["think"], "low")
        self.assertEqual(captured[0]["options"]["num_predict"], 4096)
        self.assertEqual(result["provider_metadata"]["request_policy"]["thinking"], "low")

    async def test_gateway_model_digest_and_policy_audit_are_preserved(self):
        digest = "sha256:" + "a" * 64
        def handler(request):
            return httpx.Response(200, json={"model": "test-model", "done": True,
                "message": {"content": '{"mentions":[],"complete":true}'},
                "workbench_model_audit": {"model_digest": digest, "policy_version": "fixture-policy",
                    "requested_request_sha256": "b" * 64, "effective_request_sha256": "c" * 64}})
        result = await candidates.provider([unit()], ONTOLOGY, "https://gateway.invalid", "test-model",
                                           transport=httpx.MockTransport(handler))
        self.assertEqual(result["provider_metadata"]["model_digest"], digest)
        self.assertEqual(result["provider_metadata"]["model_digest_status"], "reported_by_provider")
        self.assertEqual(result["provider_metadata"]["gateway_audits"][0]["effective_request_sha256"], "c" * 64)


class DeterministicAnchorsTests(unittest.TestCase):
    def test_later_token_block_maps_back_to_original_unit_offsets(self):
        source = "x" * candidates.LIMITS["source_block_chars"] + "臺 灣\n石拍"
        spans, _ = candidates._windows([unit(source)])
        indexed = {span["span_id"]: span for span in spans}
        register = {key: candidates._tokens(value) for key, value in indexed.items()}
        first = next(token["token_id"] for token in register["S2"] if token["text"] == "臺")
        last = next(token["token_id"] for token in register["S2"] if token["text"] == "灣")
        selected = candidates._token_mentions({"mentions": [{"span_id": "S2", "start_token": first, "end_token": last}],
                                                "complete": True}, indexed, register)
        mention = next(iter(selected.values()))
        self.assertEqual(mention["text"], "臺 灣")
        self.assertEqual(mention["start_char"], candidates.LIMITS["source_block_chars"])
        self.assertEqual(source[mention["start_char"]:mention["end_char"]], mention["text"])

    def test_overlap_keeps_english_and_chinese_boundary_names_selectable(self):
        for source, name in (("x " * 299 + "Taiwan", "Taiwan"), ("前" * 599 + "臺灣後", "臺灣")):
            spans, _ = candidates._windows([unit(source)])
            matching = [span for span in spans if name in span["text"]]
            self.assertTrue(matching)
            for span in matching:
                self.assertEqual(span["text"], source[span["start_char"]:span["end_char"]])

    def test_overlap_covers_every_source_character_within_the_window_budget(self):
        source = "樹皮布" * 500
        spans, windows = candidates._windows([unit(source)])
        covered = {i for span in spans for i in range(span["start_char"], span["end_char"])}
        self.assertEqual(covered, set(range(len(source))))
        self.assertLessEqual(len(windows), candidates.LIMITS["max_windows"])

    def test_token_ranges_preserve_traditional_glyphs_and_original_spaces(self):
        source = "前言：臺 灣  地區的樹皮布。"
        spans, _ = candidates._windows([unit(source)])
        indexed = {span["span_id"]: span for span in spans}
        register = {key: candidates._tokens(value) for key, value in indexed.items()}
        first = next(token["token_id"] for token in register["S1"] if token["text"] == "臺")
        last = next(token["token_id"] for token in register["S1"] if token["text"] == "區")
        selected = candidates._token_mentions({"mentions": [{"span_id": "S1", "start_token": first, "end_token": last}],
                                                "complete": True}, indexed, register)
        mention = next(iter(selected.values()))
        self.assertEqual(mention["text"], "臺 灣  地區")
        self.assertEqual(source[mention["start_char"]:mention["end_char"]], "臺 灣  地區")

    def test_token_selectors_reject_reversed_boolean_and_unregistered_ranges(self):
        spans, _ = candidates._windows([unit("臺灣 石拍")])
        indexed = {span["span_id"]: span for span in spans}
        register = {key: candidates._tokens(value) for key, value in indexed.items()}
        for start, end in [(0, 1), (2, 1), (1, 999), (True, 2)]:
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, "token_range_out_of_bounds"):
                candidates._token_mentions({"mentions": [{"span_id": "S1", "start_token": start, "end_token": end}],
                                            "complete": True}, indexed, register)

    def test_pdf_newlines_stay_in_one_exact_original_context_block(self):
        source = "Type IV\nbarkcloth beaters\nfound at Buttu Batu\ndated to c.\n2111–1933\nBP."
        spans, windows = candidates._windows([unit(source)])
        self.assertEqual(len(spans), 1)
        self.assertEqual(len(windows), 1)
        self.assertEqual(spans[0]["text"], source)
        self.assertEqual((spans[0]["start_char"], spans[0]["end_char"]), (0, len(source)))

    def test_ambiguous_mentions_require_explicit_occurrence(self):
        spans = {"S1": {"span_id": "S1", "unit_id": "p7", "start_char": 10, "end_char": 20,
                         "text": "石拍和石拍"}}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            candidates._mentions({"mentions": [{"span_id": "S1", "text": "石拍", "occurrence": None}], "complete": True}, spans)
        value = candidates._mentions({"mentions": [{"span_id": "S1", "text": "石拍", "occurrence": 1}], "complete": True}, spans)
        mention = next(iter(value.values()))
        self.assertEqual((mention["start_char"], mention["end_char"]), (13, 15))
        self.assertEqual(mention["text"], "石拍")

    def test_profile_identity_is_versioned_and_content_hashed(self):
        first, second = candidates.profile_identity(), candidates.profile_identity()
        self.assertEqual(first, second)
        self.assertEqual(len(first["code_sha256"]), 64)
        self.assertEqual(first["adapter_version"], candidates.ADAPTER_VERSION)


if __name__ == "__main__":
    unittest.main()
