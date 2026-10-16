"""Offline evidence-contract tests; BeautifulSoup is loaded only during tests."""

from __future__ import annotations

import unittest
import importlib.util
import hashlib
import json
from types import ModuleType
from unittest.mock import patch

if __package__:
    from .extraction import extract_html
else:
    from extraction import extract_html


URL = "https://example.org/research/page.html"


@unittest.skipUnless(importlib.util.find_spec("bs4"), "BeautifulSoup setup is pending")
class HtmlExtractionTests(unittest.TestCase):
    def original_node(self, html: str, locator: str):
        from bs4 import BeautifulSoup

        node = BeautifulSoup(html, "html.parser").select_one(locator)
        self.assertIsNotNone(node, f"Source locator did not resolve: {locator}")
        return node

    def test_noise_is_pruned_without_losing_headings_and_paragraphs(self):
        html = """<html><body><main>
          <h1>Barkcloth workshop</h1>
          <script>Untrusted script instructions</script>
          <style>Unwanted style text</style>
          <nav><p>Navigation menu</p></nav>
          <div role="navigation"><p>Secondary navigation</p></div>
          <form><p>Submit a form</p></form>
          <p>Strip the bark, then <em>soak</em> the fibres.</p>
          <footer><p>Site footer</p></footer>
        </main></body></html>"""
        result = extract_html(html, URL)
        self.assertEqual([block["tag"] for block in result["blocks"]], ["h1", "p"])
        self.assertEqual(result["text"], "Barkcloth workshop\n\nStrip the bark, then soak the fibres.")
        for unwanted in ("instructions", "style text", "navigation", "Submit", "footer"):
            self.assertNotIn(unwanted, result["text"])

    def test_table_rows_remain_citable_and_cell_paragraphs_are_not_duplicated(self):
        html = """<html><body><article><h2>Recorded activities</h2><table>
          <tr><th>Activity</th><th>Tool</th></tr>
          <tr><td><p>Beating</p></td><td><strong>Grooved stone</strong></td></tr>
        </table></article></body></html>"""
        result = extract_html(html, URL)
        rows = [block for block in result["blocks"] if block["tag"] == "tr"]
        self.assertEqual([row["text"] for row in rows], ["Activity | Tool", "Beating | Grooved stone"])
        self.assertEqual(len(result["blocks"]), 3)
        for row in rows:
            self.assertEqual(self.original_node(html, row["locator"]).name, "tr")

    def test_mixed_parent_keeps_prefix_child_and_suffix_once_in_order(self):
        html = "<main><div>Gather bark.<p>Soak fibres.</p>Beat the sheet.</div><p>Dry it.</p></main>"
        result = extract_html(html, URL)
        self.assertEqual([block["text"] for block in result["blocks"]],
                         ["Gather bark. Soak fibres. Beat the sheet.", "Dry it."])
        self.assertEqual([block["tag"] for block in result["blocks"]], ["div", "p"])
        self.assertEqual(result["text"].count("Soak fibres."), 1)
        for block in result["blocks"]:
            self.assertEqual(self.original_node(html, block["locator"]).name, block["tag"])

    def test_identical_mixed_fragments_have_one_unambiguous_owner_digest(self):
        html = "<main><div>Again.<p>Beat.</p>Again.</div></main>"
        blocks = extract_html(html, URL)["blocks"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["text"], "Again. Beat. Again.")
        self.assertEqual(blocks[0]["block_id"], hashlib.sha256(
            json.dumps([blocks[0]["text"], blocks[0]["locator"]], ensure_ascii=False).encode("utf-8")
        ).hexdigest())

    def test_bare_root_text_survives_around_nested_blocks(self):
        html = "<main>Begin.<section><p>Middle.</p></section>End.</main>"
        blocks = extract_html(html, URL)["blocks"]
        self.assertEqual([(block["tag"], block["text"]) for block in blocks],
                         [("main", "Begin. Middle. End.")])
        self.assertEqual(self.original_node(html, blocks[0]["locator"]).name, "main")
        bare = extract_html("Bare text without an element.", URL)
        self.assertEqual(bare["text"], "Bare text without an element.")
        self.assertEqual(bare["blocks"][0]["locator"], "[document]")
        self.assertTrue(any("no element CSS locator" in warning for warning in bare["warnings"]))

    def test_inline_markup_preserves_words_punctuation_and_line_breaks(self):
        html = "<main><p>Be<strong>at</strong>ing: <em>bark</em>, fibres.<br>Next <span>step</span>.</p><p>树<em>皮</em>布</p></main>"
        result = extract_html(html, URL)
        self.assertEqual([block["text"] for block in result["blocks"]],
                         ["Beating: bark, fibres. Next step.", "树皮布"])
        self.assertEqual(result["extraction_policy"], "dom_blocks_v2_owner_inline_text")

    def test_nested_lists_and_quotes_preserve_document_order_without_duplicates(self):
        html = """<main><ul><li>Prepare<ul><li>Strip bark.</li><li>Soak bark.</li></ul>Then beat.</li>
          <li>Dry sheets.</li></ul><blockquote><p>Reported by a maker.</p><p>Uncertain attribution.</p></blockquote>
          <blockquote>Quoted prefix.<p>Quoted step.</p>Quoted suffix.</blockquote></main>"""
        result = extract_html(html, URL)
        self.assertEqual([block["text"] for block in result["blocks"]],
                         ["Prepare Strip bark. Soak bark. Then beat.", "Dry sheets.",
                          "Reported by a maker.", "Uncertain attribution.",
                          "Quoted prefix. Quoted step. Quoted suffix."])
        self.assertEqual(result["text"].count("Strip bark."), 1)
        for block in result["blocks"]:
            self.original_node(html, block["locator"])

    def test_nested_tables_are_owned_by_outer_row_without_duplicate_cells(self):
        html = """<main><table><caption>Recorded materials</caption><tr><td>Outer
          <table><tr><td>Inner bark</td></tr></table>Tail</td><td>Stone beater</td></tr></table></main>"""
        result = extract_html(html, URL)
        self.assertEqual([(block["tag"], block["text"]) for block in result["blocks"]],
                         [("caption", "Recorded materials"), ("tr", "Outer Inner bark Tail | Stone beater")])
        self.assertEqual(result["text"].count("Inner bark"), 1)
        for block in result["blocks"]:
            self.assertEqual(self.original_node(html, block["locator"]).name, block["tag"])

    def test_head_noise_and_comments_do_not_enter_mixed_root_text(self):
        html = """<!doctype html><html><head><title>Metadata title</title><script>head script</script></head>
          <body><main>Keep<!-- hidden comment --><nav>menu</nav><script>instructions</script>
          <style>style text</style><noscript>fallback noise</noscript><template>template noise</template>
          <p>the bark.</p><form>form text</form><footer>footer text</footer>Finish.</main></body></html>"""
        result = extract_html(html, URL)
        self.assertEqual(result["title"], "Metadata title")
        self.assertEqual(result["text"], "Keep the bark. Finish.")
        self.assertEqual(len(result["blocks"]), 1)

    def test_pure_containers_retain_fine_blocks_and_overlapping_roots_do_not_repeat(self):
        html = "<main><div><p>First.</p><section><p>Second.</p></section></div><p>Third.</p></main>"
        result = extract_html(html, URL, content_selector="main, main div, main p")
        self.assertEqual([block["text"] for block in result["blocks"]], ["First.", "Second.", "Third."])
        self.assertEqual([block["tag"] for block in result["blocks"]], ["p", "p", "p"])

    def test_metadata_and_json_ld_are_retained_separately_from_page_text(self):
        html = """<html lang="fj"><head><title>Archive page</title>
          <meta property="og:title" content="A documented workshop">
          <meta property="article:published_time" content="2024-04-03T09:00:00Z">
          <link rel="canonical" href="/workshop">
          <script type="application/ld+json">{"@type":"Article","datePublished":"2023-01-01","name":"Archive record"}</script>
          <script type="application/ld+json">[{"@type":"Place","name":"Reported village"}]</script>
        </head><body><main><p>Documented production steps.</p></main></body></html>"""
        result = extract_html(html, URL)
        self.assertEqual(result["title"], "A documented workshop")
        self.assertEqual(result["language"], "fj")
        self.assertEqual(result["canonical_url"], "https://example.org/workshop")
        self.assertEqual(result["published_at"], "2024-04-03T09:00:00Z")
        self.assertEqual(result["structured_data"][0]["name"], "Archive record")
        self.assertEqual(result["structured_data"][1], [{"@type": "Place", "name": "Reported village"}])
        self.assertEqual(result["text"], "Documented production steps.")

    def test_nested_json_ld_publication_date_and_invalid_json(self):
        html = """<head>
          <script type="application/ld+json">{"@graph":[{"@type":"Article","datePublished":"2020-06-17"}]}</script>
          <script type="application/ld+json">This is not JSON.</script>
        </head><article><p>A field description.</p></article>"""
        result = extract_html(html, URL)
        self.assertEqual(result["published_at"], "2020-06-17")
        self.assertEqual(len(result["structured_data"]), 1)
        self.assertTrue(any("JSON-LD" in warning for warning in result["warnings"]))

    def test_selector_and_locators_use_original_dom_indices_before_pruning(self):
        html = """<html><body>
          <div role="navigation"><p>Do not cite this menu.</p></div>
          <div class="evidence"><p>Soak the fibres.</p><p>Dry the sheet.</p></div>
        </body></html>"""
        result = extract_html(html, URL, content_selector="body > div:nth-of-type(2)")
        self.assertEqual([block["text"] for block in result["blocks"]], ["Soak the fibres.", "Dry the sheet."])
        for block in result["blocks"]:
            self.assertIn("div:nth-of-type(2)", block["locator"])
            node = self.original_node(html, block["locator"])
            self.assertEqual(" ".join(node.get_text(" ", strip=True).split()), block["text"])

    def test_block_ids_are_stable_after_normalization_and_change_with_evidence(self):
        compact = "<main><p>Strip the bark.</p></main>"
        spaced = "<main><p> Strip  the\n bark. </p></main>"
        changed = "<main><p>Soak the bark.</p></main>"
        first = extract_html(compact, URL)["blocks"][0]
        repeat = extract_html(compact, URL)["blocks"][0]
        normalized = extract_html(spaced, URL)["blocks"][0]
        edited = extract_html(changed, URL)["blocks"][0]
        self.assertEqual(first["block_id"], repeat["block_id"])
        self.assertEqual(first["block_id"], normalized["block_id"])
        self.assertNotEqual(first["block_id"], edited["block_id"])
        self.assertEqual(first["locator"], edited["locator"])
        self.assertRegex(first["block_id"], r"^[0-9a-f]{64}$")

    def test_links_are_normalized_deduplicated_and_malformed_links_are_omitted(self):
        html = """<article><p>References:
          <a href=" ../tools?type=beater#spec "> Tool catalogue </a>
          <a href="../tools?type=beater#spec">Tool catalogue</a>
          <a href="//archive.example.net/photo">Museum photograph</a>
          <a href="#section">Local section</a>
          <a href="javascript:alert(1)">Script</a>
          <a href="mailto:example@example.org">Email</a>
          <a href="https://[malformed">Malformed</a>
        </p></article>"""
        result = extract_html(html, URL)
        self.assertEqual(result["links"], [
            {"url": "https://example.org/tools?type=beater#spec", "text": "Tool catalogue"},
            {"url": "https://archive.example.net/photo", "text": "Museum photograph"},
        ])
        self.assertTrue(any("malformed hyperlink" in warning for warning in result["warnings"]))

    def test_malformed_canonical_is_metadata_warning_and_keeps_requested_url(self):
        html = '<head><link rel="canonical" href="https://[malformed"></head><main><p>Evidence.</p></main>'
        result = extract_html(html, URL)
        self.assertEqual(result["canonical_url"], URL)
        self.assertEqual(result["text"], "Evidence.")
        self.assertTrue(any("Canonical metadata" in warning for warning in result["warnings"]))

    def test_invalid_selector_falls_back_to_article_and_records_warning(self):
        result = extract_html("<article><p>Retained evidence.</p></article>", URL, content_selector="[")
        self.assertEqual(result["text"], "Retained evidence.")
        self.assertTrue(any("selector" in warning.casefold() for warning in result["warnings"]))

    def test_selected_removed_noise_root_falls_back_to_retained_main(self):
        html = "<nav>Navigation text.</nav><main><p>Retained evidence.</p></main>"
        result = extract_html(html, URL, content_selector="nav")
        self.assertEqual(result["text"], "Retained evidence.")
        self.assertTrue(any("removed as document noise" in warning for warning in result["warnings"]))
        self.assertEqual(self.original_node(html, result["blocks"][0]["locator"]).name, "p")

    def test_optional_article_text_never_replaces_dom_evidence(self):
        html = "<article><h1>Workshop</h1><p>Observed bark preparation.</p></article>"
        original = extract_html(html, URL)
        supplementary = ModuleType("trafilatura")
        supplementary.extract = lambda *args, **kwargs: "Supplementary wording without source alignment."
        with patch.dict("sys.modules", {"trafilatura": supplementary}):
            enriched = extract_html(html, URL, use_trafilatura=True)
        self.assertEqual(enriched["article_text"], "Supplementary wording without source alignment.")
        self.assertEqual(enriched["blocks"], original["blocks"])
        self.assertEqual(enriched["text"], original["text"])
        self.assertEqual(enriched["extraction_method"], original["extraction_method"])


if __name__ == "__main__":
    unittest.main()
