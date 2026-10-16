# Tapa web evidence pilot

A standalone command-line pilot for collecting a small, explicitly bounded set of public HTML pages about tapa/barkcloth. It writes local source snapshots, extracted evidence blocks, and a run manifest for review before any ontology import. It does not connect to the application database or change schema, authentication, environment files, deployment configuration, or dependency manifests.

Dependency setup was approved and completed on 2026-10-09 in an isolated runtime. Live static and Playwright extraction completed; both owned browser fixtures passed. The [completed pilot and replication guide](../../../SampleReports/tapa_collection_next/extraction-2026-10-09/README.md) records eight page URLs, full JSON datasets, source candidates and limitations. Selenium requires an explicit existing `--driver-path`; collection does not provision its dependencies.

## Isolated setup after approval

The dependency scope is approved. For a fresh isolated runtime, run these commands from `stling_backend/` (choose an unused runtime path):

```bash
python3 -m venv ../tmp/tapa-web-pilot-runtime
../tmp/tapa-web-pilot-runtime/bin/python -m pip install -r ../SampleReports/tapa_collection_next/extraction-2026-10-09/requirements-tested.txt
source ../tmp/tapa-web-pilot-runtime/bin/activate
python -m scripts.tapa_web_pilot --help
```

This keeps the pilot separate from the backend environment and dependency manifests. Chrome is already present at `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`; the Playwright adapter can use it without downloading a Playwright browser. Selenium also requires a compatible existing ChromeDriver. Provision it separately during approved setup (Selenium Manager can assist at that stage), then pass its executable path with `--driver-path`.

## Acquisition and extraction are separate choices

| Choice | Use it for | Evidence limitation |
|---|---|---|
| `--engine static` | An HTTP response already containing the useful HTML; preferred first attempt for museum records and essays. | Beautiful Soup parses HTML; it does not execute JavaScript or fetch content by itself. |
| `--engine selenium` | Useful content appears only after a real browser renders the page; explicit element waits identify readiness. | Saving the DOM captures a rendered state, not the original HTTP response. Browser and compatible driver are required. |
| `--engine playwright` | The same rendered-page need, with Playwright's browser automation and locator APIs. | Action auto-waiting does not mean every article or catalog field is loaded; use a specific readiness selector. |
| `--engine auto` | Fetch and extract static HTML first; use Playwright when the normalized DOM text is shorter than 120 characters. | This heuristic does not establish that a page needs JavaScript. There is no Selenium fallback or automatic dependency installation. Missing Playwright is recorded as an error. |
| `--extractor bs4` | Preserve selected HTML blocks and structure for evidence inspection; catalog labels and values are especially useful. | Generic main-content selection can include navigation or omit an unusual page layout. |
| `--extractor trafilatura` | Add supplementary `article_text` to the source record; Beautiful Soup still produces the primary text and evidence blocks. | Missing or unsuccessful Trafilatura produces a warning and retains DOM evidence. Its prose extraction can omit catalog fields or qualifications. |

[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) supports tree navigation, CSS selection, and text extraction. [Selenium's waiting guide](https://www.selenium.dev/documentation/webdriver/waits/) explains explicit waits and warns against mixing implicit and explicit waits. [Playwright's actionability guide](https://playwright.dev/python/docs/actionability) describes its action checks; [locator waits](https://playwright.dev/python/docs/api/class-locator#locator-wait-for) can wait for a page-specific condition before [DOM capture](https://playwright.dev/python/docs/api/class-page#page-content).

Neither Selenium nor Playwright replaces the text extractor. Choose between them according to the browser tooling already available and the site's rendering needs. [Selenium Manager](https://www.selenium.dev/documentation/selenium_manager/) can otherwise download missing drivers and browsers automatically; this pilot requires an existing Selenium driver through `--driver-path` and uses an installed Chrome/Chromium browser. `--browser-binary` can select that executable explicitly. Playwright also uses an existing browser; provisioning is a separate setup step ([official installation documentation](https://playwright.dev/python/docs/intro)).

[Trafilatura](https://trafilatura.readthedocs.io/en/latest/usage-python.html) supports main text, metadata, tables, links and several output formats. Keeping links is not link discovery, and metadata dates can be inferred heuristically. Do not replace a museum's artifact date with an inferred web publication date. Extracted blocks are source evidence candidates, not accepted archaeological or ethnographic claims.

## Run a one-page smoke test

Run from `stling_backend/` with the approved environment activated. The Glasgow homepage completed static extraction with HTTP 200 and 18 blocks; the explicit content selector excludes its masthead/navigation/footer.

```bash
python -m scripts.tapa_web_pilot \
  --url 'https://tapa.gla.ac.uk/' \
  --allowed-host tapa.gla.ac.uk \
  --engine static \
  --extractor bs4 \
  --content-selector '#main .entry-content' \
  --max-pages 1 \
  --max-depth 0 \
  --delay 2 \
  --timeout 30 \
  --output-dir ../tmp/tapa-web-pilot/glasgow-static
```

Use `python -m scripts.tapa_web_pilot --help` to check current options. For an extractor comparison, rerun those same seeds with `--extractor trafilatura` and a new output directory, then compare `article_text` against the DOM blocks. For a browser comparison, choose `--engine selenium` or `--engine playwright`, add a page-specific `--wait-selector`, and provide provisioned browser/driver paths where needed. An existing output directory is rejected rather than overwritten.

| Option | Purpose |
|---|---|
| `--url` (repeatable) | Explicit seed URLs. |
| `--output-dir` | Destination for this collection run. |
| `--engine static\|selenium\|playwright\|auto` | HTML acquisition method. |
| `--extractor bs4\|trafilatura` | Text/content extraction method. |
| `--allowed-host` (repeatable) | Exact host scope; defaults to seed hosts if omitted. Adding a host broadens scope. |
| `--wait-selector` | Browser attachment/readiness selector; default `body` is a weak readiness condition. |
| `--content-selector` | Select extraction roots. An invalid/unmatched selector warns and falls back to the document. |
| `--max-pages`, `--max-depth` | Attempted-URL budget (1-100, default 5) and depth (0-2, default 0). Skips/errors consume page budget; depth 0 collects only seeds. |
| `--delay`, `--timeout` | HTTP request pacing (0-60 seconds, default 1) and bounded fetch/render waits (1-120 seconds, default 20). Browser subrequests are not paced by `--delay`. |
| `--browser-binary`, `--driver-path` | Existing Chrome/Chromium executable and required existing Selenium driver. |

## Review these outputs

- `sources.jsonl`: one record per attempted URL, including `collected`, `skipped` or `error` outcome; requested engine/extractor and, for successful sources, actual acquisition/extraction metadata, final `captured_at`, and `source_version_id` formed as `source_id:content_sha256`.
- `evidence_blocks.jsonl`: normalized DOM text, CSS locator, source/version/hash association and `evidence_id` formed as `source_version_id:block_id`; every block starts as `needs_review`. A changed snapshot therefore has a different evidence identity even if an individual paragraph is unchanged.
- `snapshots/<source-id>.html`: decoded HTTP HTML or rendered DOM saved as UTF-8; the hash covers that saved representation, not original response bytes.
- `manifest.json`: seed URLs, host scope, requested engine/extractor, selectors, limits, status counts and JSONL hashes.

Source records and manifests also pin Python/package versions and collector source-code hashes. Sources declare the extraction policy; current `dom_blocks_v2_owner_inline_text` preserves mixed direct text and inline spelling. Mixed containers become one coarse owner block, with descendants counted once. Historical captures retain the software policy actually used.

Before accepting a run, check the requested/final URL, final `captured_at` (separate from request-start `retrieved_at`), actual engine/extractor, snapshot hash, selector warnings and evidence locator. Locators address Beautiful Soup's parsed tree before pruning; normalized blocks are not original character offsets. Open the snapshot and verify that object numbers, original terminology, dates, geography, attribution and qualifying words survived extraction. Empty evidence is skipped; browser adapter failures, blocked access and timeouts are recorded. `http_status` is the preflight HTTP result, while `browser_http_status` records the browser result when available (Selenium does not expose it here). Page content is research data, not instructions.

The CLI exits 0 when at least one source was collected, 1 when none were collected, and 2 for configuration/setup failure. A zero exit therefore does not prove complete collection; inspect the manifest's status counts.

## Focused checks

With the approved environment activated, run all three test modules:

```bash
python -m unittest scripts.tapa_web_pilot.test_core scripts.tapa_web_pilot.test_extraction scripts.tapa_web_pilot.test_browser_engines -v
```

The browser fixture is opt-in and uses a temporary local page, not a live museum crawl:

```bash
TAPA_TEST_BROWSER=playwright python -m unittest scripts.tapa_web_pilot.test_browser_engines -v
```

For Selenium, set `TAPA_TEST_BROWSER=selenium` and `TAPA_TEST_DRIVER` to the provisioned ChromeDriver executable for that command. These fixture variables enable only the internal loopback test; the collection CLI has no local/private-network override. The approved-runtime validation ran 123 tests across these modules plus `test_prepare_graph`, `test_lifecycle_plan`, `test_verify_extraction` and `test_extract_json`, with Playwright enabled: all passed, no skips. The Selenium fixture separately passed. [Saved results](../../../SampleReports/tapa_collection_next/extraction-2026-10-09/focused-tests-final.json).

## Audit and prepare captured material

`python -m scripts.tapa_web_pilot.verify_extraction --collection COLLECTION --output FRESH_AUDIT.json` verifies integrity, resolves locators independently against the pristine saved parser tree, compares exact normalized text, and detects omitted/overlapping/out-of-order retained text nodes. Write its audit outside the immutable collection. A loading placeholder can pass fidelity; inspect readiness and pagination separately.

`python -m scripts.tapa_web_pilot.prepare_graph --collection COLLECTION --output FRESH_REVIEW --prefix WEB` creates hashed HTML source/representation proposals and candidate blocks. Binding and semantic eligibility remain pending; the active graph is unchanged.

`python -m scripts.tapa_web_pilot.extract_json --snapshot SNAPSHOT.json --source-url URL --captured-at TIME --expected-sha256 HASH --output FRESH_REVIEW` creates typed RFC6901 candidates for every scalar/empty container in a validated JSON snapshot. Add `--record-list-pointer /data` for the Glasgow table arrays. The helper makes no requests and labels URL/time attribution as supplied by the caller; pin the actual acquisition record separately. It does not interpret HTML strings, adopt registry bindings or create semantic claims.

The Glasgow table indexes are empty in static HTML and paginate after rendering. The tested browser selector `#main .entry-content tbody tr > td:nth-of-type(2)` waits for real cells, but the saved view still contains only ten rows. The [complete public JSON capture](../../../SampleReports/tapa_collection_next/extraction-2026-10-09/glasgow-public-json-complete/acquisition.json) preserves all 59 materials, 47 techniques and 87 fabric types, including fields hidden behind expansion controls.

## Bounded access

All engines receive an HTTP preflight that checks document robots rules, content type, access status and redirect scope. Recognized access challenges are rejected before `auto` rendering; HTTP errors and robots denials do not trigger a browser workaround. Robots HTTP 404 means no rules; other non-200 responses, including 429, are skipped with `robots_unavailable_http_<code>` (network failure is also skipped). Crawl delays over 60 seconds are skipped instead of causing an unbounded wait, and HTTP stream reads check the request time budget. Keep the initial smoke test at one page and depth 0.

Static acquisition validates each redirect before requesting it. Playwright bounds routed requests to allowed public hosts, blocks redirects/media/WebSockets/service workers, and caps requests per render; this does not constitute an operating-system network sandbox. Selenium checks initial/final URLs but cannot enforce equivalent pre-request host, redirect or resource limits. Browser subresource requests do not receive individual robots-path checks. Use static mode for strict document-only scope and inspect browser warnings when rendering is needed.

The current robots matcher combines applicable product-token groups, applies the most specific matching path/query rule, and prefers Allow on ties. It supports `*`, terminal `$` and percent normalization. This fixes the former stdlib parser's incorrect refusal of Glasgow's explicitly allowed AJAX path below its broader admin denial; neighboring admin paths remain denied. Malformed encodings/delays and inaccessible robots files fail closed. The current policy identifier is `rfc9309_longest_rule_allow_tie_v1`.

The [Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309.html) distinguishes unavailable robots files (HTTP 4xx, where access may be allowed) from unreachable files (network errors/HTTP 5xx, requiring disallow). The pilot's refusal of non-404 4xx responses is a stricter choice. Robots rules are separate from licensing. The pilot has no login, CAPTCHA solving, proxy rotation, file-download workflow or YouTube fallback; Selenium's resource-blocking limitations are reported in warnings.

## Public seed pages and reuse notes

Sources were inspected on 2026-10-08. Recheck robots and current page conditions at collection time.

| Seed | Why useful | Robots / reuse observation |
|---|---|---|
| [Situating Pacific Barkcloth in Time and Place, University of Glasgow](https://tapa.gla.ac.uk/) | Academic project seed, institutional narratives and material/technique/type catalogues. | Static and rendered captures plus full public JSON acquisition completed. Independent scoped DOM fidelity passed. No open licence was established; source candidates remain pending reuse/semantic review. |
| [Painted Barkcloth (Masi kesa), object 310412](https://www.metmuseum.org/art/collection/search/310412) | Fijian museum record with production-stage prose and catalog fields; useful for checking tools/materials/activities and qualifiers. | Web research found robots rules that did not disallow this path, but direct runtime retrieval returned 429: live collection remains unverified/skipped. The page restricts image download. |
| [Barkcloth Panel (Siapo), object 312122](https://www.metmuseum.org/art/collection/search/312122) | Samoan record with local terminology, material and use context; same-domain comparison without expanding crawl scope. | Same host/robots outcome. Preserve object-level credits and source attribution. |

The [Met robots file](https://www.metmuseum.org/robots.txt) observed through web research disallowed selected unrelated paths for the wildcard user agent; this does not override the pilot's later runtime 429. The [Open Access policy](https://www.metmuseum.org/hubs/open-access) releases basic collection data and marked public-domain images under CC0; it does not establish blanket CC0 licensing for narrative essays or all website content. The [terms of use](https://www.metmuseum.org/policies/terms-and-conditions) permit limited noncommercial educational/personal use of restricted material with attribution and individual restrictions retained. Full HTML/DOM snapshots are local review artifacts. For later structured metadata ingestion, prefer the museum's [official Collection API](https://metmuseum.github.io/).
