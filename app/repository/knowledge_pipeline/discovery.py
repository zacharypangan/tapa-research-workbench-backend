"""Bounded scholarly URL discovery. Registry metadata is never source evidence.

Uses the public Crossref works endpoint; no credentials, invented URLs, browser
sessions or recursively inherited seed attribution. Collection remains subject to
its existing public-address, robots, snapshot and exact-text checks.
"""
from __future__ import annotations

import hashlib
import asyncio
import ipaddress
import re
import unicodedata
from urllib.parse import quote, urlsplit

import httpx

from .store import now, write_json

DISCOVERY_ENDPOINT = "https://api.crossref.org/works"
MAX_DISCOVERY_QUERIES = 5


def _normal(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def public_url(value) -> str | None:
    if not isinstance(value, str) or len(value) > 4096:
        return None
    try:
        url = urlsplit(value)
    except ValueError:
        return None
    host = (url.hostname or "").casefold().rstrip(".")
    if url.scheme not in {"https", "http"} or not host or url.username or url.password or url.fragment:
        return None
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        return None
    return value


def _rank(title: str, seed: dict, topic: str) -> float:
    title = _normal(title)
    labels = [str(seed.get("label", "")), str(seed.get("label_en", ""))] + list(seed.get("aliases", []))
    score = 0.0
    for label in labels:
        label = _normal(label)
        if not label:
            continue
        tokens = [token for token in re.findall(r"\w+", label) if len(token) >= 3]
        matched = sum(token in title for token in tokens) / max(1, len(tokens))
        score = max(score, 4.0 if label in title else matched)
    # Topic overlap alone cannot turn an unrelated result into a seed-bound page.
    if score <= 0:
        return 0.0
    topic_tokens = [token for token in re.findall(r"\w+", _normal(topic)) if len(token) >= 3]
    return score + .5 * sum(token in title for token in topic_tokens) / max(1, len(topic_tokens))


async def discover_frontier(seeds: list[dict], topic: str, output_path, max_pages: int) -> dict:
    if type(max_pages) is not int or not 1 <= max_pages <= 25:
        raise ValueError("Discovery requires a 1–25-page hard budget")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Discovery requires document-derived seeds")
    audit, candidates = [], []
    selected_seeds = sorted(seeds, key=lambda seed: (len(seed.get("reported_relationship_ids", [])), seed["seed_id"]))[:MAX_DISCOVERY_QUERIES]
    rows = min(5, max_pages)
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0), follow_redirects=False) as client:
        for index, seed in enumerate(selected_seeds):
            if index:
                await asyncio.sleep(.5)
            query = " ".join(filter(None, [seed.get("label_en") or seed["label"], topic]))[:500]
            attempt = {"seed_id": seed["seed_id"], "query": query, "endpoint": DISCOVERY_ENDPOINT, "at": now()}
            try:
                response = await client.get(DISCOVERY_ENDPOINT, params={"query": query, "rows": rows},
                                            headers={"User-Agent": "ResearchWorkbench/1.1 (bounded scholarly discovery)"})
                attempt["http_status"] = response.status_code
                if response.status_code != 200 or len(response.content) > 2 * 1024 * 1024:
                    attempt["status"] = "unavailable"
                    audit.append(attempt)
                    continue
                payload = response.json()
                attempt.update(status="queried", response=payload,
                               response_sha256=hashlib.sha256(response.content).hexdigest())
                items = payload.get("message", {}).get("items", [])
                if not isinstance(items, list):
                    raise ValueError("Discovery response omitted items")
                for item in items[:rows]:
                    if not isinstance(item, dict):
                        continue
                    title = " ".join(value for value in item.get("title", []) if isinstance(value, str)) if isinstance(item.get("title"), list) else ""
                    rank = _rank(title, seed, topic)
                    if not rank:
                        continue
                    resource = item.get("resource", {})
                    primary_data = resource.get("primary", {}) if isinstance(resource, dict) else {}
                    primary = primary_data.get("URL") if isinstance(primary_data, dict) else None
                    url = public_url(primary) or public_url(item.get("URL"))
                    doi = item.get("DOI")
                    if url is None and isinstance(doi, str) and re.match(r"^10\.\d{4,9}/\S+$", doi):
                        url = "https://doi.org/" + quote(doi, safe="/")
                    if url is None:
                        continue
                    candidates.append({"url": url, "seed_ids": [seed["seed_id"]], "question": seed["question"],
                                       "discovery_method": "crossref_title_ranked", "query": query,
                                       "discovery_locator": DISCOVERY_ENDPOINT, "discovery_title": title,
                                       "discovery_score": rank, "discovery_doi": doi})
            except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
                attempt.update(status="unavailable", error_type=type(exc).__name__)
            audit.append(attempt)
    merged = {}
    for candidate in sorted(candidates, key=lambda item: (-item["discovery_score"], item["url"])):
        if candidate["url"] in merged:
            old = merged[candidate["url"]]
            old["seed_ids"] = sorted(set(old["seed_ids"] + candidate["seed_ids"]))
        else:
            merged[candidate["url"]] = candidate
    frontier = list(merged.values())[:max_pages]
    result = {"method": "crossref_title_ranked-v1", "frontier": frontier, "attempts": audit,
              "max_queries": MAX_DISCOVERY_QUERIES, "max_pages": max_pages,
              "interpretation": "Discovery metadata ranks candidate URLs, not evidence or a relevance guarantee."}
    write_json(output_path, result)
    return result
