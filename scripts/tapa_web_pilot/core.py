"""Bounded HTTP collection, robots checks, and evidence exports."""

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
from urllib.parse import urldefrag, urljoin, urlsplit

import httpx

from .browser_engines import USER_AGENT, render_html, validate_public_url
from .extraction import extract_html


MAX_BYTES = 3_000_000
REDIRECT_CODES = {301, 302, 303, 307, 308}
ROBOTS_POLICY = "rfc9309_longest_rule_allow_tie_v1"
_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_URI_ASCII = _UNRESERVED | frozenset("/?:@!&'()+,;=*$")


def _robots_octets(value, *, pattern=False):
    """Canonicalize UTF-8 and percent escapes without decoding reserved bytes.

    Raw path/query delimiters retain their meaning. Encoded reserved bytes stay
    encoded, so /a%2Fb is distinct from /a/b. Only pattern '*' and final '$' are
    operators; URI occurrences of these characters are literal encoded bytes.
    """
    result, index = [], 0
    while index < len(value):
        char = value[index]
        if char == "%" and re.fullmatch(r"[0-9A-Fa-f]{2}", value[index + 1:index + 3]):
            byte = int(value[index + 1:index + 3], 16)
            decoded = chr(byte)
            result.append(decoded if decoded in _UNRESERVED else f"%{byte:02X}")
            index += 3
            continue
        if char == "%":
            raise ValueError("Malformed percent escape in robots path")
        if char not in _URI_ASCII or (
                char in "*$" and not (pattern and (char == "*" or index == len(value) - 1))):
            result.extend(f"%{byte:02X}" for byte in char.encode("utf-8"))
        else:
            result.append(char)
        index += 1
    return "".join(result)


def _robots_match(pattern, target):
    """Match an anchored prefix with '*' and optional terminal '$' without regex backtracking."""
    anchored = pattern.endswith("$")
    if anchored:
        pattern = pattern[:-1]
    parts = pattern.split("*")
    if not target.startswith(parts[0]):
        return False
    cursor = len(parts[0])
    if len(parts) == 1:
        return not anchored or cursor == len(target)
    for index, part in enumerate(parts[1:], 1):
        if anchored and index == len(parts) - 1:
            return len(target) - len(part) >= cursor and target.endswith(part)
        found = target.find(part, cursor)
        if found < 0:
            return False
        cursor = found + len(part)
    return True


class RobotsRules:
    """RFC 9309 path precedence, group merging and wildcard matching.

    Exact case-insensitive product-token groups are combined; '*' is the fallback.
    Malformed path percent escapes and crawl delays fail closed through Fetcher.
    Crawl-delay is an extension: retain the greatest valid declared group delay.
    HTTP availability and safe redirect policy remain the caller's responsibility.
    """

    def __init__(self):
        self.groups = []

    def parse(self, lines):
        self.groups = []
        current, has_rules = None, False
        for raw in lines:
            line = raw.lstrip("\ufeff").split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key = key.casefold()
            if key == "user-agent":
                if current is None or has_rules:
                    current = {"agents": [], "rules": [], "delays": []}
                    self.groups.append(current)
                    has_rules = False
                if value == "*" or re.fullmatch(r"[A-Za-z_-]+", value):
                    current["agents"].append(value.casefold())
            elif current is not None and key in {"allow", "disallow"}:
                has_rules = True
                if not value or not value.startswith(("/", "*")):
                    continue
                normalized = _robots_octets(value, pattern=True)
                # Wildcards and terminal anchors carry no literal match specificity.
                specificity = len(normalized.rstrip("$").replace("*", "").encode("ascii"))
                current["rules"].append((normalized, specificity, key == "allow"))
            elif current is not None and key == "crawl-delay":
                has_rules = True
                delay = float(value)
                if not math.isfinite(delay) or delay < 0:
                    raise ValueError("Invalid robots crawl-delay")
                current["delays"].append(delay)
            # Other records, including Sitemap, do not terminate groups (RFC 9309 2.2.4).

    def _groups(self, user_agent):
        token = user_agent.split("/", 1)[0].casefold()
        matched = [group for group in self.groups if token != "*" and token in group["agents"]]
        return matched or [group for group in self.groups if "*" in group["agents"]]

    def can_fetch(self, user_agent, url):
        parts = urlsplit(url)
        if parts.path == "/robots.txt":
            return True
        target = parts.path or "/"
        if "?" in urldefrag(url)[0]:
            target += "?" + parts.query
        target = _robots_octets(target)
        matched = [(length, allowed) for group in self._groups(user_agent)
                   for pattern, length, allowed in group["rules"] if _robots_match(pattern, target)]
        return max(matched)[1] if matched else True

    def crawl_delay(self, user_agent):
        delays = [delay for group in self._groups(user_agent) for delay in group["delays"]]
        return max(delays) if delays else None


@dataclass
class Config:
    url: list[str]
    output_dir: Path
    engine: str = "static"
    extractor: str = "bs4"
    allowed_host: list[str] = field(default_factory=list)
    content_selector: str | None = None
    wait_selector: str = "body"
    max_pages: int = 5
    max_depth: int = 0
    delay: float = 1.0
    timeout: float = 20.0
    browser_binary: str | None = None
    driver_path: str | None = None


class CollectionFailure(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class Fetcher:
    def __init__(self, config, hosts):
        self.config, self.hosts = config, hosts
        self.robots = {}
        self.last_request = 0.0
        self.client = httpx.Client(
            timeout=config.timeout, follow_redirects=False, trust_env=False,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        )

    def close(self):
        self.client.close()

    def request(self, url, extra_delay=0):
        validate_public_url(url, self.hosts)
        pause = max(self.config.delay, extra_delay) - (time.monotonic() - self.last_request)
        if pause > 0:
            time.sleep(pause)
        self.last_request = time.monotonic()
        deadline = self.last_request + self.config.timeout
        try:
            with self.client.stream("GET", url) as response:
                body = bytearray()
                for chunk in response.iter_bytes():
                    if time.monotonic() > deadline:
                        raise CollectionFailure("response_time_limit")
                    body.extend(chunk)
                    if len(body) > MAX_BYTES:
                        raise CollectionFailure("response_too_large")
                return response.status_code, dict(response.headers), bytes(body), response.encoding
        except httpx.HTTPError:
            raise CollectionFailure("network_error") from None

    def robots_for(self, url):
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin in self.robots:
            return self.robots[origin]
        current = origin + "/robots.txt"
        try:
            for _ in range(6):
                status, headers, body, _ = self.request(current)
                if status in REDIRECT_CODES and headers.get("location"):
                    current = urljoin(current, headers["location"])
                    continue
                parser = RobotsRules()
                if status == 404:
                    parser.parse([])
                elif status == 200:
                    parser.parse(body.decode("utf-8", errors="replace").splitlines())
                else:
                    raise CollectionFailure(f"robots_unavailable_http_{status}")
                self.robots[origin] = parser
                return parser
        except ValueError:
            raise CollectionFailure("robots_unavailable") from None
        except CollectionFailure as exc:
            if exc.reason.startswith("robots_"):
                raise
            raise CollectionFailure("robots_unavailable_" + exc.reason) from None
        raise CollectionFailure("robots_redirect_limit")

    def permit(self, url):
        validate_public_url(url, self.hosts)
        rules = self.robots_for(url)
        if not rules.can_fetch(USER_AGENT, url):
            raise CollectionFailure("robots_disallowed")
        delay = rules.crawl_delay(USER_AGENT)
        if delay is None:
            delay = 0
        if delay > 60:
            raise CollectionFailure("robots_crawl_delay_exceeds_budget")
        return delay

    def html(self, url):
        chain = []
        for _ in range(6):
            delay = self.permit(url)
            status, headers, body, encoding = self.request(url, delay)
            chain.append({"url": url, "status": status})
            if status in REDIRECT_CODES and headers.get("location"):
                url = urljoin(url, headers["location"])
                continue
            if status in {401, 403, 429}:
                raise CollectionFailure(f"access_restricted_http_{status}")
            if status != 200:
                raise CollectionFailure(f"http_{status}")
            media_type = headers.get("content-type", "").split(";", 1)[0].lower()
            if media_type not in {"text/html", "application/xhtml+xml"}:
                raise CollectionFailure("unsupported_content_type")
            try:
                html = body.decode(encoding or "utf-8", errors="replace")
            except LookupError:
                html = body.decode("utf-8", errors="replace")
            return html, url, status, chain
        raise CollectionFailure("redirect_limit")


def hash_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def collection_software():
    """Pin the actual extraction implementation and runtime without reading env."""
    packages = {}
    for name in ("httpx", "beautifulsoup4", "soupsieve", "selenium", "playwright", "trafilatura"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    directory = Path(__file__).parent
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "parser": "BeautifulSoup/html.parser", "packages": packages,
            "robots_policy": ROBOTS_POLICY,
            "code_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                            for name in ("__main__.py", "core.py", "extraction.py", "browser_engines.py")}}


def reject_challenge(html, parsed):
    if "cf-chl-" in html.lower() or (parsed.get("title") or "").lower() in {
        "access denied", "just a moment...",
    }:
        raise CollectionFailure("access_challenge")


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    path.chmod(0o600)


def collect(config):
    """Collect into a new output directory without touching repository databases."""
    hosts = set(config.allowed_host or [urlsplit(url).hostname for url in config.url])
    hosts.discard(None)
    seeds = [urldefrag(validate_public_url(url, hosts))[0] for url in config.url]
    if config.output_dir.exists():
        raise ValueError("Choose a new output directory; existing outputs are not replaced")
    # Fail before creating outputs if the required static parser is unavailable.
    import bs4  # noqa: F401

    software = collection_software()
    config.output_dir.mkdir(parents=True, mode=0o700)
    snapshots = config.output_dir / "snapshots"
    snapshots.mkdir(mode=0o700)
    queue = deque((url, 0, None) for url in seeds)
    seen, hashes, sources, evidence = set(), {}, [], []
    fetcher = Fetcher(config, hosts)
    try:
        while queue and len(seen) < config.max_pages:
            requested, depth, discovered_from = queue.popleft()
            if requested in seen:
                continue
            seen.add(requested)
            retrieved = datetime.now(timezone.utc).isoformat()
            source_id = "web-" + hash_text(requested)[:20]
            record = {"source_id": source_id, "requested_url": requested, "retrieved_at": retrieved,
                      "depth": depth, "discovered_from": discovered_from, "review_status": "needs_review",
                      "requested_engine": config.engine, "requested_extractor": config.extractor,
                      "software": software}
            try:
                # Fetch first to validate robots, content type, redirects and access status.
                html, final, status, chain = fetcher.html(requested)
                parsed = extract_html(html, final, config.content_selector, config.extractor == "trafilatura")
                reject_challenge(html, parsed)
                engine = config.engine
                if engine == "auto":
                    engine = "playwright" if len(parsed["text"]) < 120 else "static"
                warnings = []
                browser_status = None
                if engine != "static":
                    rendered = render_html(final, engine, wait_selector=config.wait_selector,
                                           timeout_seconds=config.timeout, allowed_hosts=hosts,
                                           browser_binary=config.browser_binary, driver_path=config.driver_path)
                    html, final = rendered["html"], rendered["final_url"]
                    fetcher.permit(final)
                    warnings = rendered.get("warnings", [])
                    browser_status = rendered.get("status_code")
                    if browser_status not in {None, 200}:
                        raise CollectionFailure(f"browser_http_{browser_status}")
                    parsed = extract_html(html, final, config.content_selector, config.extractor == "trafilatura")
                reject_challenge(html, parsed)
                if not parsed["text"].strip():
                    raise CollectionFailure("no_extractable_text")
                if len(html.encode("utf-8")) > MAX_BYTES:
                    raise CollectionFailure("rendered_response_too_large")
                content_hash = hash_text(html)
                source_version_id = source_id + ":" + content_hash
                captured_at = datetime.now(timezone.utc).isoformat()
                snapshot = snapshots / f"{source_id}.html"
                snapshot.write_text(html, encoding="utf-8")
                snapshot.chmod(0o600)
                record.update({"status": "collected", "final_url": final, "http_status": status,
                               "browser_http_status": browser_status,
                               "source_version_id": source_version_id, "captured_at": captured_at,
                               "engine": engine, "representation": "rendered_dom" if engine != "static" else "http_html",
                               "content_sha256": content_hash, "snapshot": str(snapshot.relative_to(config.output_dir)),
                               "duplicate_of": hashes.get(content_hash), "redirect_chain": chain,
                               "title": parsed["title"], "language": parsed["language"],
                               "canonical_url_metadata": parsed["canonical_url"], "published_at_metadata": parsed["published_at"],
                               "extraction_method": parsed["extraction_method"], "text": parsed["text"],
                               "extraction_policy": parsed.get("extraction_policy", "dom_blocks_v1"),
                               "structured_data": parsed["structured_data"], "warnings": warnings + parsed["warnings"]})
                if parsed.get("article_text"):
                    record["article_text"] = parsed["article_text"]
                    record["article_text_method"] = parsed["article_text_method"]
                hashes.setdefault(content_hash, source_id)
                evidence.extend({**block, "evidence_id": source_version_id + ":" + block["block_id"],
                                 "source_id": source_id, "source_version_id": source_version_id,
                                 "source_url": final, "captured_at": captured_at,
                                 "source_content_sha256": content_hash, "review_status": "needs_review"}
                                for block in parsed["blocks"])
                if depth < config.max_depth:
                    for link in parsed["links"]:
                        child = urldefrag(link["url"])[0]
                        if len(queue) + len(seen) >= config.max_pages * 4:
                            break
                        try:
                            validate_public_url(child, hosts)
                        except ValueError:
                            continue
                        if child not in seen:
                            queue.append((child, depth + 1, final))
            except CollectionFailure as exc:
                record.update(status="skipped", reason=exc.reason)
            except (ImportError, ValueError, RuntimeError) as exc:
                record.update(status="error", reason=type(exc).__name__, detail=str(exc)[:300])
            except Exception as exc:
                record.update(status="error", reason=type(exc).__name__)
            sources.append(record)
    finally:
        fetcher.close()
    write_jsonl(config.output_dir / "sources.jsonl", sources)
    write_jsonl(config.output_dir / "evidence_blocks.jsonl", evidence)
    manifest = {"format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "software": software,
                "successful_sources": sum(row["status"] == "collected" for row in sources),
                "attempted_sources": len(sources), "evidence_blocks": len(evidence),
                "statuses": dict(Counter(row["status"] for row in sources)),
                "allowed_hosts": sorted(hosts), "requested_engine": config.engine,
                "requested_extractor": config.extractor, "seed_urls": seeds,
                "selectors": {"content": config.content_selector, "wait": config.wait_selector},
                "limits": {"max_pages": config.max_pages, "max_depth": config.max_depth,
                           "delay": config.delay, "timeout": config.timeout, "response_bytes": MAX_BYTES},
                "files": {name: {"sha256": hashlib.sha256((config.output_dir / name).read_bytes()).hexdigest()}
                          for name in ("sources.jsonl", "evidence_blocks.jsonl")}}
    (config.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (config.output_dir / "manifest.json").chmod(0o600)
    return manifest
