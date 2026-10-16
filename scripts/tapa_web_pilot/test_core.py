"""Focused offline checks for collection limits and source provenance."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from . import core


def response(status=200, body="", content_type="text/html", **headers):
    return httpx.Response(status, text=body, headers={"content-type": content_type, **headers})


class FetcherFixture(unittest.TestCase):
    def setUp(self):
        self.urls = []
        self.responses = {"/robots.txt": response(body="User-agent: *\nAllow: /\n", content_type="text/plain")}
        self.guard = patch.object(core, "validate_public_url", side_effect=self.validate)
        self.guard.start()
        self.fetcher = core.Fetcher(core.Config(["https://example.org/"], Path("unused"), delay=0), {"example.org"})
        self.fetcher.client.close()
        self.fetcher.client = httpx.Client(transport=httpx.MockTransport(self.handle))

    def tearDown(self):
        self.fetcher.close()
        self.guard.stop()

    @staticmethod
    def validate(url, hosts):
        if core.urlsplit(url).hostname not in hosts:
            raise ValueError("Outside allowed hosts")
        return url

    def handle(self, request):
        self.urls.append(str(request.url))
        return self.responses.get(request.url.path, response(404))

    def assert_failure(self, reason, url="https://example.org/page"):
        with self.assertRaises(core.CollectionFailure) as caught:
            self.fetcher.html(url)
        self.assertEqual(caught.exception.reason, reason)


class FetcherTests(FetcherFixture):
    def test_most_specific_allow_beats_broad_disallow_in_either_order(self):
        for lines in (("Disallow: /wp-admin/", "Allow: /wp-admin/admin-ajax.php"),
                      ("Allow: /wp-admin/admin-ajax.php", "Disallow: /wp-admin/")):
            with self.subTest(lines=lines):
                rules = core.RobotsRules()
                rules.parse(["User-agent: *", *lines])
                self.assertTrue(rules.can_fetch(core.USER_AGENT, "https://example.org/wp-admin/admin-ajax.php"))
                self.assertTrue(rules.can_fetch(core.USER_AGENT, "https://example.org/wp-admin/admin-ajax.php?action=public"))
                for path in ("/wp-admin/", "/wp-admin/index.php", "/wp-admin/admin-ajax-other.php"):
                    self.assertFalse(rules.can_fetch(core.USER_AGENT, "https://example.org" + path))

    def test_ajax_request_passes_permission_and_more_specific_query_stays_denied(self):
        self.responses["/robots.txt"] = response(body="User-agent: *\nDisallow: /wp-admin/\n"
                                                "Allow: /wp-admin/admin-ajax.php\n"
                                                "Disallow: /wp-admin/admin-ajax.php?action=private\n")
        self.responses["/wp-admin/admin-ajax.php"] = response(body="<p>Public catalogue</p>")
        public = "https://example.org/wp-admin/admin-ajax.php?action=public&item=20"
        self.assertEqual(self.fetcher.html(public)[0], "<p>Public catalogue</p>")
        private = "https://example.org/wp-admin/admin-ajax.php?action=private&item=20"
        self.assert_failure("robots_disallowed", private)
        self.assertNotIn(private, self.urls)

    def test_equivalent_allow_wins_tie_and_path_case_is_sensitive(self):
        rules = core.RobotsRules()
        rules.parse(["USER-AGENT: *", "Allow: /same", "Disallow: /same", "Disallow: /Case"])
        self.assertTrue(rules.can_fetch(core.USER_AGENT, "https://example.org/same"))
        self.assertFalse(rules.can_fetch(core.USER_AGENT, "https://example.org/Case"))
        self.assertTrue(rules.can_fetch(core.USER_AGENT, "https://example.org/case"))

    def test_exact_product_groups_merge_and_other_records_do_not_end_group(self):
        rules = core.RobotsRules()
        rules.parse(["Disallow: /orphan", "User-agent: *", "Disallow: /", "User-agent: taparesearchpilot",
                     "Disallow: /private", "Sitemap: https://example.org/map.xml", "Allow: /private/public",
                     "User-agent: OtherBot", "Disallow: /other", "User-agent: TapaResearchPilot",
                     "Disallow: /second", "Crawl-delay: 2", "User-agent: TapaResearchPilot", "Crawl-delay: 5"])
        for path in ("/private", "/second"):
            self.assertFalse(rules.can_fetch(core.USER_AGENT, "https://example.org" + path))
        for path in ("/private/public", "/other", "/orphan"):
            self.assertTrue(rules.can_fetch(core.USER_AGENT, "https://example.org" + path))
        self.assertEqual(rules.crawl_delay(core.USER_AGENT), 5)
        self.assertFalse(rules.can_fetch("UnknownBot/1.0", "https://example.org/page"))

    def test_wildcard_end_anchor_comments_and_literal_special_characters(self):
        rules = core.RobotsRules()
        rules.parse(["User-agent: *", "Disallow: /private/*", "Allow: /private/public$ # comment",
                     "Disallow: /*.gif$", "Disallow: /literal-%2A", "Disallow: /price-%24"])
        self.assertTrue(rules.can_fetch(core.USER_AGENT, "https://example.org/private/public#fragment"))
        for path in ("/private/public/child", "/private/public?x=1", "/images/file.gif", "/literal-*", "/price-$"):
            self.assertFalse(rules.can_fetch(core.USER_AGENT, "https://example.org" + path))
        self.assertTrue(rules.can_fetch(core.USER_AGENT, "https://example.org/images/file.gif?size=small"))
        self.assertTrue(rules.can_fetch(core.USER_AGENT, "https://example.org/images/file.gif.jpg"))

    def test_percent_normalization_utf8_and_reserved_octets(self):
        rules = core.RobotsRules()
        rules.parse(["User-agent: *", "Disallow: /private", "Disallow: /café", "Disallow: /encoded%2Fpath", "Disallow: /unsafe%3C"])
        for path in ("/%70rivate", "/caf%C3%A9", "/café", "/encoded%2fpath", "/unsafe<"):
            self.assertFalse(rules.can_fetch(core.USER_AGENT, "https://example.org" + path))
        self.assertTrue(rules.can_fetch(core.USER_AGENT, "https://example.org/encoded/path"))
        encoded_rule = core.RobotsRules()
        encoded_rule.parse(["User-agent: *", "Disallow: /%70rivate"])
        self.assertFalse(encoded_rule.can_fetch(core.USER_AGENT, "https://example.org/private"))

    def test_empty_query_marker_and_empty_group_rules_are_handled(self):
        rules = core.RobotsRules()
        rules.parse(["User-agent: *", "Disallow:", "Allow:", "Disallow: /query?$", "Disallow: /"])
        self.assertFalse(rules.can_fetch(core.USER_AGENT, "https://example.org/query?"))
        self.assertTrue(rules.can_fetch(core.USER_AGENT, "https://example.org/robots.txt"))
        rules.parse(["User-agent: *", "Disallow: /", "User-agent: TapaResearchPilot"])
        self.assertTrue(rules.can_fetch(core.USER_AGENT, "https://example.org/page"))

    def test_malformed_percent_rule_and_invalid_delays_fail_closed(self):
        for directive in ("Disallow: /bad%Q0", "Crawl-delay: nan", "Crawl-delay: -1"):
            with self.subTest(directive=directive):
                self.fetcher.robots.clear()
                self.responses["/robots.txt"] = response(body="User-agent: *\n" + directive + "\n")
                self.assert_failure("robots_unavailable")
        self.assertTrue(all(url.endswith("/robots.txt") for url in self.urls))

    def test_non404_client_robots_errors_still_fail_closed(self):
        for status in (403, 429):
            with self.subTest(status=status):
                self.fetcher.robots.clear()
                self.responses["/robots.txt"] = response(status)
                self.assert_failure(f"robots_unavailable_http_{status}")
        self.assertTrue(all(url.endswith("/robots.txt") for url in self.urls))

    def test_fractional_robots_delay_above_budget_stays_denied(self):
        self.responses["/robots.txt"] = response(body="User-agent: *\nCrawl-delay: 60.01\n")
        self.assert_failure("robots_crawl_delay_exceeds_budget")
        self.assertEqual(self.urls, ["https://example.org/robots.txt"])

    def test_selected_product_delay_zero_or_absent_does_not_use_wildcard_delay(self):
        for directive in ("Crawl-delay: 0\n", ""):
            with self.subTest(directive=directive):
                self.fetcher.robots.clear()
                self.responses["/robots.txt"] = response(body=
                    "User-agent: *\nCrawl-delay: 61\nAllow: /\n"
                    "User-agent: TapaResearchPilot\nAllow: /\n" + directive)
                self.assertEqual(self.fetcher.permit("https://example.org/page"), 0)
        self.fetcher.robots.clear()
        self.responses["/robots.txt"] = response(body="User-agent: *\nCrawl-delay: 61\nAllow: /\n")
        self.assert_failure("robots_crawl_delay_exceeds_budget")
        self.assertTrue(all(url.endswith("/robots.txt") for url in self.urls))

    def test_robots_disallow_does_not_request_page(self):
        self.responses["/robots.txt"] = response(body="User-agent: *\nDisallow: /page\n")
        self.assert_failure("robots_disallowed")
        self.assertEqual(self.urls, ["https://example.org/robots.txt"])

    def test_unavailable_robots_fails_closed(self):
        self.responses["/robots.txt"] = response(503)
        self.assert_failure("robots_unavailable_http_503")
        self.assertEqual(len(self.urls), 1)

    def test_missing_robots_allows_html(self):
        self.responses.update({"/robots.txt": response(404), "/page": response(body="<p>Tapa</p>")})
        self.assertEqual(self.fetcher.html("https://example.org/page")[0], "<p>Tapa</p>")

    def test_redirect_checks_new_target_robots(self):
        self.responses.update({
            "/robots.txt": response(body="User-agent: *\nDisallow: /private\n"),
            "/page": response(302, location="/private"),
        })
        self.assert_failure("robots_disallowed")
        self.assertNotIn("https://example.org/private", self.urls)

    def test_external_redirect_is_not_requested(self):
        self.responses["/page"] = response(302, location="https://other.org/page")
        with self.assertRaises(ValueError):
            self.fetcher.html("https://example.org/page")
        self.assertFalse(any("other.org" in url for url in self.urls))

    def test_access_restrictions_and_media_are_explicit(self):
        for status in (401, 403, 429):
            with self.subTest(status=status):
                self.responses["/page"] = response(status)
                self.assert_failure(f"access_restricted_http_{status}")
        self.responses["/page"] = response(content_type="application/pdf")
        self.assert_failure("unsupported_content_type")

    def test_response_size_is_bounded(self):
        self.responses["/page"] = response(body="x" * 65)
        with patch.object(core, "MAX_BYTES", 64):
            self.assert_failure("response_too_large")

    def test_slow_stream_has_an_overall_time_budget(self):
        self.responses["/page"] = response(body="Text")
        with patch.object(core.time, "monotonic", side_effect=[100, 100, 121]):
            with self.assertRaises(core.CollectionFailure) as caught:
                self.fetcher.request("https://example.org/page")
        self.assertEqual(caught.exception.reason, "response_time_limit")

    def test_excessive_robots_delay_skips_page(self):
        self.responses["/robots.txt"] = response(body="User-agent: *\nCrawl-delay: 90\n")
        self.assert_failure("robots_crawl_delay_exceeds_budget")
        self.assertEqual(self.urls, ["https://example.org/robots.txt"])

    def test_challenge_detection_accepts_missing_title(self):
        core.reject_challenge("<p>Fine</p>", {"title": None})
        with self.assertRaises(core.CollectionFailure):
            core.reject_challenge("<p>cf-chl-token</p>", {"title": None})


@unittest.skipUnless(importlib.util.find_spec("bs4"), "BeautifulSoup setup is pending")
class CollectionTests(FetcherFixture):
    def test_static_pipeline_keeps_evidence_and_hashes(self):
        html = '<html lang="en"><main><h1>Tapa</h1><p>Bark is beaten into cloth.</p><a href="/next">Next</a></main></html>'
        self.responses.update({"/page": response(body=html), "/next": response(body=html)})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "collected"
            config = core.Config(["https://example.org/page"], output, max_pages=2, max_depth=1, delay=0)
            with patch.object(core, "Fetcher", return_value=self.fetcher):
                manifest = core.collect(config)
            records = [json.loads(line) for line in (output / "sources.jsonl").read_text().splitlines()]
            blocks = [json.loads(line) for line in (output / "evidence_blocks.jsonl").read_text().splitlines()]
            self.assertEqual(manifest["successful_sources"], 2)
            self.assertEqual(records[1]["duplicate_of"], records[0]["source_id"])
            self.assertEqual(records[1]["discovered_from"], "https://example.org/page")
            self.assertTrue(all(block["review_status"] == "needs_review" for block in blocks))
            self.assertTrue(all(block["evidence_id"].startswith(block["source_version_id"] + ":") for block in blocks))
            self.assertEqual(len({block["evidence_id"] for block in blocks}), len(blocks))
            for record in records:
                snapshot = (output / record["snapshot"]).read_bytes()
                self.assertEqual(hashlib.sha256(snapshot).hexdigest(), record["content_sha256"])
                self.assertEqual(record["software"], manifest["software"])
                self.assertEqual(record["extraction_policy"], "dom_blocks_v2_owner_inline_text")
                self.assertEqual(record["software"]["robots_policy"], core.ROBOTS_POLICY)
            self.assertEqual(manifest["software"]["code_sha256"]["extraction.py"],
                             hashlib.sha256(Path(core.__file__).with_name("extraction.py").read_bytes()).hexdigest())
            for filename, metadata in manifest["files"].items():
                self.assertEqual(hashlib.sha256((output / filename).read_bytes()).hexdigest(), metadata["sha256"])
            self.assertEqual((output / "sources.jsonl").stat().st_mode & 0o777, 0o600)

    def test_page_limit_and_depth_zero(self):
        self.responses["/page"] = response(body='<main><p>Text</p><a href="/next">Next</a></main>')
        with tempfile.TemporaryDirectory() as directory:
            config = core.Config(["https://example.org/page"], Path(directory) / "out", max_pages=1, delay=0)
            with patch.object(core, "Fetcher", return_value=self.fetcher):
                manifest = core.collect(config)
            self.assertEqual(manifest["attempted_sources"], 1)
            self.assertNotIn("https://example.org/next", self.urls)

    def test_auto_does_not_render_access_challenge(self):
        self.responses["/page"] = response(body='<title>Just a moment...</title>')
        with tempfile.TemporaryDirectory() as directory:
            config = core.Config(["https://example.org/page"], Path(directory) / "out", engine="auto", delay=0)
            with patch.object(core, "Fetcher", return_value=self.fetcher), patch.object(core, "render_html") as render:
                manifest = core.collect(config)
            render.assert_not_called()
            record = json.loads((config.output_dir / "sources.jsonl").read_text())
            self.assertEqual(record["reason"], "access_challenge")
            self.assertEqual(manifest["successful_sources"], 0)

    def test_empty_page_is_reported(self):
        self.responses["/page"] = response(body='<html><body><script>1</script></body></html>')
        with tempfile.TemporaryDirectory() as directory:
            config = core.Config(["https://example.org/page"], Path(directory) / "out", delay=0)
            with patch.object(core, "Fetcher", return_value=self.fetcher):
                core.collect(config)
            self.assertEqual(json.loads((config.output_dir / "sources.jsonl").read_text())["reason"], "no_extractable_text")

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / "keep.txt"
            sentinel.write_text("Keep me")
            with self.assertRaises(ValueError):
                core.collect(core.Config(["https://example.org/page"], output))
            self.assertEqual(sentinel.read_text(), "Keep me")


if __name__ == "__main__":
    unittest.main()
