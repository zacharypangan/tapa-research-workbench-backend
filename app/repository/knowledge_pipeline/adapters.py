"""Source-preserving ingestion adapters; no database or semantic inference.

Files are written exclusively into fresh directories. Extracted text is a
representation, not a claim that a source is complete, legible or scientifically
accepted. Optional PDF/browser libraries are imported only when requested.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import re
from urllib.parse import urldefrag, urlsplit
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile


MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
MAX_DOCX_EXPANDED_BYTES = 100 * 1024 * 1024
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = "{" + WORD_NS + "}"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _identifier(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(f"{label} must be a nonempty identifier of at most 200 characters")
    if any(ord(c) < 32 for c in value):
        raise ValueError(f"{label} contains control characters")
    return value


def _fresh_directory(path: Path) -> Path:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Choose a fresh output directory; existing ingestion is immutable")
    try:
        path.mkdir(parents=True, mode=0o700)
    except FileExistsError as exc:
        raise ValueError("Output directory already exists") from exc
    return path.resolve()


def _write(path: Path, data: bytes):
    with path.open("xb") as handle:
        handle.write(data)
    path.chmod(0o400)


def _seal_files(directory: Path):
    for file in directory.rglob("*"):
        if file.is_file():
            file.chmod(0o400)


def _unit(source_id: str, version: str, locator: dict, text: str,
          seed_ids: list[str] | None = None) -> dict:
    text_hash = _sha(text.encode("utf-8"))
    identity = _sha(_json_bytes([locator, text_hash]))
    return {"unit_id": version + ":unit:" + identity,
            "source_id": source_id, "source_version_id": version,
            "locator": locator, "text": text, "text_sha256": text_hash,
            "seed_ids": list(seed_ids or [])}


def _utf8(data: bytes, label: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must contain valid UTF-8 text; encoding was not guessed") from exc


def _text_parts(text: str) -> list[tuple[dict, str]]:
    parts = []
    for match in re.finditer(r"(?:[^\n]|\n(?!\s*\n))+", text):
        raw = match.group()
        value = raw.strip()
        if not value:
            continue
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.end() - (len(raw) - len(raw.rstrip()))
        parts.append(({"kind": "text_span", "char_start": start, "char_end": end,
                       "offset_basis": "UTF-8-decoded text, optional BOM removed"}, value))
    return parts


def _reject_constant(value):
    raise ValueError("JSON must not contain nonfinite values: " + value)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains duplicate key: " + key)
        result[key] = value
    return result


def _json_parts(data: bytes) -> list[tuple[dict, str]]:
    try:
        value = json.loads(_utf8(data, "JSON"), object_pairs_hook=_unique_pairs,
                           parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed JSON document") from exc
    parts = []

    def visit(item, pointer):
        if isinstance(item, dict) and item:
            for key, child in item.items():
                token = key.replace("~", "~0").replace("/", "~1")
                visit(child, pointer + "/" + token)
        elif isinstance(item, list) and item:
            for index, child in enumerate(item):
                visit(child, pointer + "/" + str(index))
        else:
            kind = ("null" if item is None else "boolean" if isinstance(item, bool)
                    else "string" if isinstance(item, str) else "object" if isinstance(item, dict)
                    else "array" if isinstance(item, list) else "number")
            text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, allow_nan=False)
            parts.append(({"kind": "json_leaf", "json_pointer": pointer,
                           "value_type": kind, "text_basis": "parsed JSON scalar or empty container"}, text))

    visit(value, "")
    return parts


def _docx_parts(data: bytes) -> tuple[list[tuple[dict, str]], list[str]]:
    try:
        with ZipFile(BytesIO(data)) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > MAX_DOCX_EXPANDED_BYTES:
                raise ValueError("DOCX expanded contents exceed the ingestion limit")
            try:
                xml = archive.read("word/document.xml")
            except KeyError as exc:
                raise ValueError("DOCX has no word/document.xml") from exc
        root = ElementTree.fromstring(xml)
    except (BadZipFile, ElementTree.ParseError, RuntimeError) as exc:
        raise ValueError("Malformed DOCX archive or main XML document") from exc
    body = root.find(W + "body")
    if body is None:
        raise ValueError("DOCX main XML document has no body")
    parts = []

    def visit(container, xpath, table_path):
        counts = defaultdict(int)
        for child in container:
            counts[child.tag] += 1
            local = child.tag.rsplit("}", 1)[-1]
            current = xpath + f"/w:{local}[{counts[child.tag]}]"
            if child.tag == W + "p":
                text = "".join(node.text or "" if node.tag == W + "t"
                               else "\t" if node.tag == W + "tab"
                               else "\n" if node.tag in {W + "br", W + "cr"} else ""
                               for node in child.iter())
                if text.strip():
                    parts.append(({"kind": "docx_paragraph", "xml_part": "word/document.xml",
                                   "xpath": current, "table_path": list(table_path)}, text))
            elif child.tag == W + "tbl":
                for row_index, row in enumerate(child.findall(W + "tr"), 1):
                    for cell_index, cell in enumerate(row.findall(W + "tc"), 1):
                        cell_path = current + f"/w:tr[{row_index}]/w:tc[{cell_index}]"
                        coordinates = {"table_index": counts[child.tag],
                                       "row": row_index, "cell": cell_index}
                        visit(cell, cell_path, table_path + [coordinates])
            elif child.tag == W + "sdt":
                content = child.find(W + "sdtContent")
                if content is not None:
                    visit(content, current + "/w:sdtContent[1]", table_path)

    visit(body, "/w:document/w:body", [])
    return parts, ["Extracted main-body XML paragraphs and table cells only; headers, footnotes and images are not analyzed."]


def _pdf_parts(data: bytes) -> tuple[list[tuple[dict, str]], str, list[str], list[dict]]:
    failures, available = [], False
    for module_name in ("fitz", "pypdf"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        available = True
        try:
            if module_name == "fitz":
                document = module.open(stream=data, filetype="pdf")
                try:
                    texts = [document.load_page(i).get_text("text") or ""
                             for i in range(document.page_count)]
                finally:
                    document.close()
            else:
                document = module.PdfReader(BytesIO(data))
                texts = [page.extract_text() or "" for page in document.pages]
            parts, empty = [], []
            for index, text in enumerate(texts, 1):
                locator = {"kind": "pdf_page", "pdf_page": index,
                           "text_basis": "parser text; layout/table interpretation not established"}
                if text.strip():
                    parts.append((locator, text))
                else:
                    empty.append({"kind": "pdf_page", "pdf_page": index})
            warnings = failures + (["Some PDF pages have no extracted text; they may be scanned or empty. No absence claim is established."] if empty else [])
            return parts, module_name, warnings, empty
        except Exception as exc:
            # Discard partial parser output before trying a fallback.
            failures.append(f"{module_name} could not read this PDF ({type(exc).__name__}).")
    if not available:
        raise ImportError("PDF ingestion requires an installed PyMuPDF (fitz) or pypdf runtime; no dependency was installed")
    raise ValueError("PDF could not be read by the available parsers: " + " ".join(failures))


def ingest_document(path: Path, source_id: str, title: str, output_dir: Path,
                    origin: str = "primary") -> dict:
    """Copy and extract a local source, preserving exact bytes and real locators."""
    path = Path(path)
    _identifier(source_id, "source_id")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be nonempty")
    _identifier(origin, "origin")
    if not path.is_file():
        raise ValueError("Document path must identify a readable file")
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise ValueError("Document exceeds the 50 MiB ingestion limit")
    suffix = path.suffix.lower()
    if suffix not in {".txt", ".md", ".json", ".docx", ".pdf"}:
        raise ValueError("Supported document formats are TXT, Markdown, JSON, DOCX and PDF")
    data = path.read_bytes()
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("Document exceeds the 50 MiB ingestion limit")
    raw_hash = _sha(data)
    warnings, empty = [], []
    if suffix in {".txt", ".md"}:
        parts, extractor, source_kind = _text_parts(_utf8(data, "Text document")), "stdlib_utf8_text_spans", "file_text"
    elif suffix == ".json":
        parts, extractor, source_kind = _json_parts(data), "stdlib_json_leaf_pointers", "file_json"
    elif suffix == ".docx":
        parts, warnings = _docx_parts(data)
        extractor, source_kind = "stdlib_docx_xml_paragraphs_tables", "file_docx"
    else:
        parts, extractor, warnings, empty = _pdf_parts(data)
        source_kind = "file_pdf"
    version = source_id + ":" + raw_hash
    units = [_unit(source_id, version, locator, text) for locator, text in parts]
    representation = _jsonl_bytes(units)
    directory = _fresh_directory(output_dir)
    snapshot = directory / ("snapshot" + suffix)
    representation_path = directory / "units.jsonl"
    _write(snapshot, data)
    _write(representation_path, representation)
    status = "unreadable_or_empty" if not units else "partial_text" if empty else "extracted"
    if not units:
        warnings.append("No readable text was extracted; this does not establish that the source lacks relevant information.")
    source = {"source_id": source_id, "source_version_id": version, "title": title,
              "origin": origin, "source_kind": source_kind, "source_locator": str(path.resolve()),
              "raw_sha256": raw_hash, "snapshot_sha256": raw_hash,
              "representation_sha256": _sha(representation),
              "snapshot_path": str(snapshot), "representation_path": str(representation_path),
              "acquisition_method": "immutable_local_file_copy", "extractor": extractor,
              "extractor_code_sha256": _sha(Path(__file__).read_bytes()),
              "python_version": platform.python_version(), "status": status,
              "unreadable_locators": empty, "warnings": warnings,
              "review_status": "needs_review", "seed_ids": []}
    _write(directory / "source.json", _json_bytes(source))
    return {"source": source, "units": units}


def _pilot_module(name: str):
    for prefix in ("scripts.tapa_web_pilot", "stling_backend.scripts.tapa_web_pilot"):
        try:
            return importlib.import_module(prefix + "." + name)
        except ModuleNotFoundError as exc:
            if exc.name and (prefix + "." + name).startswith(exc.name):
                continue
            raise ImportError(f"Web collection requires the existing {exc.name} runtime; no dependency was installed") from exc
    raise ImportError("The existing tapa_web_pilot collector is not importable in this runtime")


def _frontier(frontier: list[dict]) -> list[dict]:
    if not isinstance(frontier, list) or not frontier:
        raise ValueError("frontier must be a nonempty list")
    entries, indexes = [], {}
    for row in frontier:
        if not isinstance(row, dict):
            raise ValueError("Each frontier entry must be an object")
        url = row.get("url")
        if not isinstance(url, str):
            raise ValueError("Each frontier entry requires a URL")
        url = urldefrag(url.strip())[0]
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Frontier URLs must be public HTTP(S) locators without credentials")
        seeds = row.get("seed_ids")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("Every frontier URL requires explicit nonempty seed_ids")
        seeds = list(dict.fromkeys(_identifier(value, "seed_id") for value in seeds))
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Every frontier URL requires its seed question")
        selectors = row.get("selectors")
        if isinstance(selectors, list):
            if not selectors or any(not isinstance(value, str) or not value.strip() for value in selectors):
                raise ValueError("selectors must be nonempty CSS strings")
            selectors = ", ".join(selectors)
        if selectors is not None and (not isinstance(selectors, str) or not selectors.strip()):
            raise ValueError("selectors must be a CSS string or a list of CSS strings")
        if url in indexes:
            entry = entries[indexes[url]]
            if entry["selectors"] != selectors:
                raise ValueError("A repeated URL has conflicting selectors")
            entry["seed_ids"] = list(dict.fromkeys(entry["seed_ids"] + seeds))
            if question not in entry["questions"]:
                entry["questions"].append(question)
        else:
            indexes[url] = len(entries)
            entries.append({"url": url, "seed_ids": seeds, "questions": [question],
                            "selectors": selectors, "planned_order": len(entries) + 1})
    return entries


def collect_frontier(frontier: list[dict], output_dir: Path, options: dict) -> dict:
    """Collect explicitly paper-seeded URLs; discovery never inherits seed links."""
    entries = _frontier(frontier)
    if not isinstance(options, dict):
        raise ValueError("options must be an object")
    permitted = {"engine", "extractor", "allowed_host", "allowed_hosts", "content_selector",
                 "wait_selector", "delay", "timeout", "max_pages", "max_depth"}
    if set(options) - permitted:
        raise ValueError("Unsupported collection options; executable paths are not accepted")
    if type(options.get("max_depth", 0)) is not int or options.get("max_depth", 0) != 0:
        raise ValueError("Paper-seeded collection requires max_depth=0; register discovered URLs explicitly")
    engine, extractor = options.get("engine", "static"), options.get("extractor", "bs4")
    if not isinstance(engine, str) or not isinstance(extractor, str) or engine not in {"static", "selenium", "playwright", "auto"} or extractor not in {"bs4", "trafilatura"}:
        raise ValueError("Unknown collection engine or extractor")
    for label in ("content_selector", "wait_selector"):
        if label in options and (not isinstance(options[label], str) or not options[label].strip()):
            raise ValueError(label + " must be a nonempty CSS string")
    limit = options.get("max_pages", len(entries))
    if type(limit) is not int or not 1 <= limit <= 100 or len(entries) > limit:
        raise ValueError("max_pages must be 1–100 and cover the explicit frontier")
    delay, timeout = options.get("delay", 1.0), options.get("timeout", 20.0)
    for value, minimum, maximum, label in [(delay, 0, 60, "delay"), (timeout, 1, 120, "timeout")]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{label} is outside the permitted finite range")
    hosts = options.get("allowed_hosts", options.get("allowed_host"))
    if hosts is None:
        hosts = sorted({urlsplit(row["url"]).hostname for row in entries})
    if not isinstance(hosts, list) or not hosts or any(not isinstance(host, str) or not host.strip() for host in hosts):
        raise ValueError("allowed_hosts must be a list of hostnames")
    if not {urlsplit(row["url"]).hostname for row in entries}.issubset(set(hosts)):
        raise ValueError("Every frontier host must be explicitly allowed")
    core, verifier = _pilot_module("core"), _pilot_module("verify_extraction")
    directory = _fresh_directory(output_dir)
    _write(directory / "seed_manifest.json", _json_bytes({"frontier": entries, "options": options,
           "max_depth": 0, "unbound_discovery_policy": "no inherited seed attribution"}))
    # Contiguous groups retain declared source order while allowing per-URL selectors.
    groups = []
    for entry in entries:
        selector = entry["selectors"] or options.get("content_selector")
        if not groups or groups[-1][0] != selector:
            groups.append((selector, []))
        groups[-1][1].append(entry)
    sources, units, attempts = [], [], []
    for index, (selector, group) in enumerate(groups, 1):
        collection = directory / f"collection-{index:03d}"
        try:
            core.collect(core.Config(url=[row["url"] for row in group], output_dir=collection,
                         engine=engine, extractor=extractor, allowed_host=hosts,
                         content_selector=selector, wait_selector=options.get("wait_selector", "body"),
                         max_pages=len(group), max_depth=0, delay=delay, timeout=timeout))
            audit = verifier.audit_collection(collection)
        except ImportError as exc:
            raise ImportError("Web collection requires the installed BeautifulSoup/httpx or explicitly requested browser runtime; no dependency was installed") from exc
        finally:
            _seal_files(collection)
        _write(directory / f"dom_audit-{index:03d}.json", _json_bytes(audit))
        if audit["status"] not in {"passed", "no_collected_sources"}:
            raise ValueError("Saved snapshot DOM fidelity check failed; no semantic ingestion is permitted")
        rows = [json.loads(line) for line in (collection / "sources.jsonl").read_text().splitlines() if line]
        blocks = [json.loads(line) for line in (collection / "evidence_blocks.jsonl").read_text().splitlines() if line]
        planned = {row["url"]: row for row in group}
        if len(rows) != len(group) or {row["requested_url"] for row in rows} != set(planned):
            raise ValueError("Collection contains an unbound discovered URL or an unaccounted frontier attempt")
        for row in rows:
            seed = planned[row["requested_url"]]
            attempt = {"source_id": row["source_id"], "url": row["requested_url"],
                       "seed_ids": seed["seed_ids"], "questions": seed["questions"],
                       "planned_order": seed["planned_order"], "status": row["status"],
                       "reason": row.get("reason"), "collection_path": str(collection)}
            attempts.append(attempt)
            if row["status"] != "collected":
                continue
            snapshot = (collection / row["snapshot"]).resolve()
            if not snapshot.is_relative_to(collection.resolve()) or not snapshot.is_file():
                raise ValueError("Snapshot locator escapes the collection or is missing")
            snapshot_hash = _sha(snapshot.read_bytes())
            if snapshot_hash != row["content_sha256"]:
                raise ValueError("Saved snapshot hash differs from its source record")
            local = [_unit(row["source_id"], row["source_version_id"],
                           {"kind": "web_dom_block", "css_selector": block["locator"],
                            "block_id": block["block_id"], "source_url": row["final_url"]},
                           block["text"], seed["seed_ids"])
                     for block in blocks if block["source_id"] == row["source_id"]]
            representation = _jsonl_bytes(local)
            representation_path = directory / (row["source_id"] + ".units.jsonl")
            _write(representation_path, representation)
            source = {"source_id": row["source_id"], "source_version_id": row["source_version_id"],
                      "title": row.get("title") or row["final_url"], "origin": "web",
                      "source_kind": "captured_web_page", "source_locator": row["final_url"],
                      "requested_url": row["requested_url"], "raw_sha256": None,
                      "snapshot_sha256": snapshot_hash, "representation_sha256": _sha(representation),
                      "snapshot_path": str(snapshot), "representation_path": str(representation_path),
                      "acquisition_method": row["representation"], "extractor": row["extraction_method"],
                      "snapshot_hash_basis": "UTF-8 saved decoded HTML or rendered DOM; original HTTP bytes not retained",
                      "captured_at": row["captured_at"], "seed_ids": seed["seed_ids"],
                      "questions": seed["questions"], "review_status": "needs_review",
                      "status": "extracted" if local else "unreadable_or_empty",
                      "warnings": row.get("warnings", []), "collection_path": str(collection)}
            sources.append(source)
            units.extend(local)
    attempts.sort(key=lambda row: row["planned_order"])
    _write(directory / "sources.jsonl", _jsonl_bytes(sources))
    _write(directory / "units.jsonl", _jsonl_bytes(units))
    _write(directory / "attempts.jsonl", _jsonl_bytes(attempts))
    _seal_files(directory)
    return {"sources": sources, "units": units, "attempts": attempts,
            "seed_manifest_path": str(directory / "seed_manifest.json")}
