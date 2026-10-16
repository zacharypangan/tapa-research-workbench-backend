"""Local HTML extraction with source DOM blocks; never fetch referenced URLs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from urllib.parse import urljoin, urlsplit


BLOCK_TAGS = frozenset({"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "pre", "blockquote", "dt", "dd", "figcaption", "caption"})
CANDIDATE_TAGS = BLOCK_TAGS | {"div", "section"}
TEXT_BOUNDARIES = CANDIDATE_TAGS | {"main", "article", "body", "html", "ul", "ol", "dl", "table", "thead", "tbody", "tfoot", "td", "th", "figure", "br", "hr"}
NOISE_TAGS = ("head", "script", "style", "nav", "footer", "form", "noscript", "template")
MAX_JSON_LD_SCRIPTS = 20
MAX_JSON_LD_CHARS = 500_000
MAX_BLOCKS = 2_000
MAX_LINKS = 500
EXTRACTION_POLICY = "dom_blocks_v2_owner_inline_text"


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _dom_paths(soup) -> dict[int, str]:
    """Capture indices before pruning so a locator still selects the source node."""
    paths: dict[int, str] = {}
    sibling_counts: dict[tuple[int, str], int] = defaultdict(int)
    for tag in soup.find_all(True):
        key = (id(tag.parent), tag.name)
        sibling_counts[key] += 1
        step = f"{tag.name}:nth-of-type({sibling_counts[key]})"
        parent = paths.get(id(tag.parent))
        paths[id(tag)] = f"{parent} > {step}" if parent else step
    return paths


def _structured_data(soup, warnings: list[str]) -> list:
    records: list = []
    used = 0
    scripts = [tag for tag in soup.find_all("script") if str(tag.get("type", "")).split(";", 1)[0].strip().casefold() == "application/ld+json"]
    if len(scripts) > MAX_JSON_LD_SCRIPTS:
        warnings.append("JSON-LD script count exceeded the extraction limit.")
    for tag in scripts[:MAX_JSON_LD_SCRIPTS]:
        raw = tag.string if tag.string is not None else tag.get_text()
        used += len(raw)
        if used > MAX_JSON_LD_CHARS:
            warnings.append("JSON-LD character budget exceeded; remaining scripts omitted.")
            break
        try:
            value = json.loads(raw)
        except (ValueError, TypeError, RecursionError):
            warnings.append("A JSON-LD script could not be parsed as JSON.")
            continue
        if isinstance(value, (dict, list)):
            records.append(value)
    return records


def _meta_content(soup, names: set[str]) -> str | None:
    for tag in soup.find_all("meta"):
        name = str(tag.get("property") or tag.get("name") or tag.get("itemprop") or "").casefold()
        content = tag.get("content")
        if name in names and isinstance(content, str) and content.strip():
            return _normalize(content)
    return None


def _published_at(soup, data: list) -> str | None:
    value = _meta_content(soup, {"article:published_time", "datepublished", "pubdate", "dc.date.issued", "dcterms.issued"})
    if value:
        return value
    pending = deque(data)
    for _ in range(1_000):
        if not pending:
            break
        item = pending.popleft()
        if isinstance(item, dict):
            value = item.get("datePublished")
            if isinstance(value, str) and value.strip():
                return _normalize(value)
            pending.extend(value for value in item.values() if isinstance(value, (dict, list)))
        elif isinstance(item, list):
            pending.extend(item[:1_000])
    for tag in soup.find_all("time", itemprop="datePublished"):
        if tag.get("datetime"):
            return _normalize(str(tag["datetime"]))
    return None


def _roots(soup, selector: str | None, warnings: list[str]) -> list:
    roots = []
    if selector:
        try:
            roots = soup.select(selector)
        except Exception as exc:
            warnings.append(f"Content selector could not be evaluated ({type(exc).__name__}).")
        if not roots:
            warnings.append("Content selector matched no retained element; using document fallback.")
    if not roots:
        root = soup.find("main") or soup.find("article") or soup.body or soup
        roots = [root]
    selected = {id(root) for root in roots}
    return [root for root in roots if not any(id(parent) in selected for parent in root.parents)]


def _ordered_text(tag) -> str:
    """Keep inline characters contiguous and separate block/line boundaries."""
    from bs4.element import Comment, Declaration, Doctype, NavigableString, ProcessingInstruction

    parts, pending = [], [(tag, False)]
    while pending:
        node, closing = pending.pop()
        if closing:
            parts.append(" ")
        elif isinstance(node, NavigableString):
            if not isinstance(node, (Comment, Declaration, Doctype, ProcessingInstruction)):
                parts.append(str(node))
        else:
            boundary = node.name in TEXT_BOUNDARIES
            if boundary:
                parts.append(" ")
                pending.append((None, True))
            pending.extend((child, False) for child in reversed(node.contents))
    return _normalize("".join(parts))


def _block_text(tag) -> str:
    if tag.name == "tr":
        cells = tag.find_all(["th", "td"], recursive=False)
        if cells:
            values = [_ordered_text(cell) for cell in cells]
            return " | ".join(values) if any(values) else ""
    return _ordered_text(tag)


def _candidate_children(tag):
    """Return the nearest candidate descendants in document order."""
    children, pending = [], list(reversed(tag.contents))
    while pending:
        node = pending.pop()
        if getattr(node, "name", None) in CANDIDATE_TAGS:
            children.append(node)
        elif getattr(node, "contents", None):
            pending.extend(reversed(node.contents))
    return children


def _has_own_text(tag):
    """Find meaningful text outside any descendant candidate block."""
    from bs4.element import Comment, Declaration, Doctype, NavigableString, ProcessingInstruction

    pending = list(reversed(tag.contents))
    while pending:
        node = pending.pop()
        if isinstance(node, NavigableString):
            if not isinstance(node, (Comment, Declaration, Doctype, ProcessingInstruction)) and _normalize(str(node)):
                return True
        elif node.name not in CANDIDATE_TAGS:
            pending.extend(reversed(node.contents))
    return False


def _source_blocks(root):
    """Use one owner for mixed text; never emit its descendants a second time.

    A CSS selector cannot address a text-node fragment. Keeping a mixed container
    whole preserves prefix/child/suffix order and keeps the existing block-ID
    contract unambiguous, including repeated identical prefix/suffix text.
    """
    pending = [root]
    while pending:
        tag = pending.pop()
        children = [] if tag.name == "tr" else _candidate_children(tag)
        if not children or _has_own_text(tag):
            yield tag
        else:
            pending.extend(reversed(children))


def extract_html(
    html: str,
    url: str,
    content_selector: str | None = None,
    use_trafilatura: bool = False,
) -> dict:
    """Extract normalized evidence blocks, not original character offsets.

    Canonical URLs and JSON-LD are reported metadata. They are never followed.
    CSS locators address the original BeautifulSoup-parsed DOM before pruning.
    Mixed containers are one block, so direct text and descendants retain their
    order without duplicate child blocks. A bare document has no element locator.
    """
    if not isinstance(html, str) or not isinstance(url, str):
        raise TypeError("html and url must be strings")
    if content_selector is not None and not isinstance(content_selector, str):
        raise TypeError("content_selector must be a string or None")
    from bs4 import BeautifulSoup

    warnings: list[str] = []
    soup = BeautifulSoup(html, "html.parser")
    paths = _dom_paths(soup)
    data = _structured_data(soup, warnings)
    title = _meta_content(soup, {"og:title"})
    if not title and soup.title:
        title = _normalize(soup.title.get_text(" ", strip=True))
    language = _normalize(str(soup.html.get("lang", ""))) or None if soup.html else None
    canonical = url
    for link in soup.find_all("link"):
        rel = link.get("rel", [])
        rel = rel.split() if isinstance(rel, str) else rel
        if "canonical" in {str(value).casefold() for value in rel} and link.get("href"):
            try:
                canonical = urljoin(url, str(link["href"]).strip())
            except ValueError:
                warnings.append("Canonical metadata contains a malformed URL.")
            break
    published = _published_at(soup, data)
    roots = _roots(soup, content_selector, warnings)
    for tag in list(soup.find_all(NOISE_TAGS)):
        tag.decompose()
    for tag in list(soup.select('[role="navigation"]')):
        tag.decompose()
    roots = [root for root in roots if root.name]
    if not roots:
        warnings.append("Selected content was removed as document noise; using document fallback.")
        roots = _roots(soup, None, warnings)
    if not title:
        heading = next((root.find("h1") for root in roots if root.find("h1")), None)
        title = _normalize(heading.get_text(" ", strip=True)) if heading else None
    blocks: list[dict] = []
    for root in roots:
        for tag in _source_blocks(root):
            text = _block_text(tag)
            if not text:
                continue
            if len(blocks) >= MAX_BLOCKS:
                warnings.append("DOM block count exceeded the extraction limit.")
                break
            locator = paths.get(id(tag), "[document]")
            if locator == "[document]":
                warnings.append("Document-level text has no element CSS locator; inspect the full snapshot.")
            digest = hashlib.sha256(json.dumps([text, locator], ensure_ascii=False).encode("utf-8")).hexdigest()
            blocks.append({"block_id": digest, "text": text, "locator": locator, "tag": tag.name})
        if len(blocks) >= MAX_BLOCKS:
            break
    links: list[dict] = []
    seen_links: set[tuple[str, str]] = set()
    for root in roots:
        anchors = ([root] if root.name == "a" and root.get("href") else []) + root.find_all("a", href=True)
        for anchor in anchors:
            href = str(anchor["href"]).strip()
            if not href or href.startswith("#"):
                continue
            try:
                resolved = urljoin(url, href)
                scheme = urlsplit(resolved).scheme
            except ValueError:
                warnings.append("A malformed hyperlink was omitted.")
                continue
            if scheme not in {"http", "https"}:
                continue
            text = _normalize(anchor.get_text(" ", strip=True))
            key = (resolved, text)
            if key in seen_links:
                continue
            if len(links) >= MAX_LINKS:
                warnings.append("Link count exceeded the extraction limit.")
                break
            seen_links.add(key)
            links.append({"url": resolved, "text": text})
        if len(links) >= MAX_LINKS:
            break
    result = {
        "title": title,
        "language": language,
        "canonical_url": canonical,
        "published_at": published,
        "text": "\n\n".join(block["text"] for block in blocks),
        "blocks": blocks,
        "links": links,
        "structured_data": data,
        "extraction_method": "beautifulsoup_dom_blocks",
        "extraction_policy": EXTRACTION_POLICY,
        "warnings": warnings,
    }
    if use_trafilatura:
        try:
            import trafilatura
        except ImportError:
            warnings.append("Optional Trafilatura is unavailable; DOM evidence blocks retained.")
        else:
            try:
                article_text = trafilatura.extract(html, url=url, include_tables=True)
                result["article_text"] = article_text
                result["article_text_method"] = "trafilatura_supplementary"
                if not article_text:
                    warnings.append("Trafilatura returned no supplementary article text.")
            except Exception as exc:
                warnings.append(f"Trafilatura supplementary extraction failed ({type(exc).__name__}); DOM evidence blocks retained.")
    return result
