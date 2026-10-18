"""Content-addressed source transcription shared by repository search and graphs.

Only native text and OCR enter units. Generated image descriptions never do.
Artifacts are immutable JSON, keyed by bytes, parser/engine versions and language.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile

METHOD_VERSION = "source-transcription-v1"
REPRESENTATION_VERSION = METHOD_VERSION
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_PAGES = 1000
MAX_OCR_PAGES = 200
OCR_TIMEOUT = 30


def representation_signature() -> dict:
    """Versions used by durable processing fingerprints and representation caches."""
    source = Path(__file__).read_bytes()
    adapters = Path(__file__).parent / "knowledge_pipeline" / "adapters.py"
    source += adapters.read_bytes() if adapters.exists() else b""
    packages = {}
    for package in ("PyMuPDF", "pypdf", "pytesseract", "Pillow"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "unavailable"
    return {"method_version": METHOD_VERSION, "implementation_sha256": hashlib.sha256(source).hexdigest(),
            "packages": packages}


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\x00", "")).strip()


def _languages(language: str | list[str] | None, filename: str = "") -> list[str]:
    aliases = {"en": "eng", "english": "eng", "zh": "chi_sim+chi_tra", "chinese": "chi_sim+chi_tra",
               "zh-cn": "chi_sim", "zh-hans": "chi_sim", "zh-tw": "chi_tra", "zh-hant": "chi_tra",
               "ja": "jpn", "japanese": "jpn", "中文": "chi_sim+chi_tra", "汉语": "chi_sim+chi_tra", "日本語": "jpn"}
    if language:
        values = language if isinstance(language, list) else re.split(r"[+,;/ ]+", language)
        return sorted({code for value in values if value for code in aliases.get(value.lower(), value.lower()).split("+")})
    if re.search(r"[\u3040-\u30ff]", filename):
        return ["eng", "jpn"]
    if re.search(r"[\u4e00-\u9fff]", filename):
        return ["chi_sim", "chi_tra", "eng"]
    return ["eng"]


def ocr_runtime(language: str | list[str] | None = None, filename: str = "") -> dict:
    requested = _languages(language, filename)
    try:
        import pytesseract
        installed = sorted(pytesseract.get_languages(config=""))
        version = str(pytesseract.get_tesseract_version()).splitlines()[0]
    except Exception as exc:
        return {"requested": requested, "selected": [], "engine_version": "unavailable",
                "warnings": [f"OCR unavailable: {type(exc).__name__}. Install Tesseract and requested language data."]}
    missing = sorted(set(requested) - set(installed))
    if not language and not re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", filename):
        # Unknown metadata/Latin filenames can still be scanned CJK documents.
        # Optional installed packs improve script coverage without requiring them.
        requested = sorted(set(requested) | (set(installed) & {"chi_sim", "chi_tra", "jpn"}))
    # Never silently substitute English for a declared Chinese/Japanese source.
    return {"requested": requested, "selected": requested if not missing else [],
            "engine_version": version, "installed": installed,
            "warnings": [f"OCR language data missing: {', '.join(missing)}; installed: {', '.join(installed)}."] if missing else []}


def ocr_image(image, runtime: dict, timeout: int = OCR_TIMEOUT) -> dict:
    warnings = list(runtime["warnings"])
    if not runtime["selected"]:
        return {"text": "", "confidence": None, "language": "+".join(runtime["requested"]), "warnings": warnings}
    try:
        import pytesseract
        data = pytesseract.image_to_data(image, lang="+".join(runtime["selected"]),
                                        timeout=timeout, output_type=pytesseract.Output.DICT)
        lines: dict[tuple, list[str]] = {}
        confidence = []
        for i, word in enumerate(data.get("text", [])):
            word = str(word).strip()
            if not word:
                continue
            key = tuple(data.get(name, [0] * len(data["text"]))[i] for name in ("block_num", "par_num", "line_num"))
            lines.setdefault(key, []).append(word)
            value = float(data.get("conf", ["-1"] * len(data["text"]))[i])
            if value >= 0:
                confidence.append(value / 100)
        text = _clean("\n".join(" ".join(words) for words in lines.values()))
        if not text:
            warnings.append("OCR completed but detected no text.")
        mean_confidence = sum(confidence) / len(confidence) if confidence else None
        if mean_confidence is not None and mean_confidence < .5:
            warnings.append("OCR confidence below 0.50; transcription needs human review.")
        return {"text": text, "confidence": mean_confidence,
                "language": "+".join(runtime["selected"]), "warnings": warnings}
    except Exception as exc:
        return {"text": "", "confidence": None, "language": "+".join(runtime["selected"]),
                "warnings": warnings + [f"OCR failed or timed out: {type(exc).__name__}: {str(exc)[:160]}"]}


def _unit(locator: str, index: int, text: str, method: str, **extra) -> dict:
    return {"locator": locator, "page_index": index, "text": text, "method": method,
            "language": None, "confidence": None, **extra}


def _write_json(path: Path, value: dict, *, immutable: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parse_pdf(path: Path, runtime: dict, page_cache: Path | None = None) -> tuple[list, list, dict]:
    units, warnings, unreadable = [], [], []
    ocr_attempted = 0
    import fitz
    from PIL import Image
    with fitz.open(path) as document:
        page_count = len(document)
        if page_count > MAX_PAGES:
            warnings.append(f"Page budget reached: reading first {MAX_PAGES} of {page_count} pages.")
        for index in range(min(page_count, MAX_PAGES)):
            page = document[index]
            text = (page.get_text("text") or "").strip()
            method, confidence, language = "native_pdf", None, None
            # Native text wins unless a text-poor page yields a richer transcription.
            needs_ocr = len(text) < 80 and bool(page.get_images(full=True) or page.get_drawings())
            if needs_ocr:
                cached_page = page_cache / f"page-{index + 1}.json" if page_cache is not None else None
                result = None
                if cached_page is not None and cached_page.exists():
                    try:
                        result = json.loads(cached_page.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        pass
                if result is None and ocr_attempted >= MAX_OCR_PAGES:
                    result = {"text": "", "warnings": [f"OCR page budget reached ({MAX_OCR_PAGES}); retry resumes uncached pages."]}
                elif result is None and runtime["selected"]:
                    ocr_attempted += 1
                    scale = min(2.0, (12_000_000 / max(page.rect.width * page.rect.height, 1)) ** .5)
                    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                    with Image.open(io.BytesIO(pix.tobytes("png"))) as image:
                        result = ocr_image(image, runtime)
                    if result["text"] and cached_page is not None:
                        _write_json(cached_page, result, immutable=True)
                elif result is None:
                    result = {"text": "", "warnings": runtime["warnings"]}
                warnings.extend(f"page:{index + 1}: {warning}" for warning in result["warnings"])
                if len(result["text"]) > len(text):
                    text, method = result["text"], "tesseract_ocr"
                    confidence, language = result["confidence"], result["language"]
            if text:
                units.append(_unit(f"page:{index + 1}", index + 1, text, method,
                                   language=language, confidence=confidence))
            else:
                unreadable.append(f"page:{index + 1}")
        return units, warnings, {"total_units": page_count, "readable_units": len(units),
                                 "ocr_attempted": ocr_attempted, "unreadable_locators": unreadable,
                                 "omitted_units": max(0, page_count - MAX_PAGES)}


def extract_document(path: str | Path, filename: str, *, cache_root=None, languages=None) -> dict:
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Document exceeds the 100 MiB transcription budget.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    extension = Path(filename).suffix.lower()
    runtime = ocr_runtime(languages, filename) if extension == ".pdf" else {"engine_version": "unused"}
    identity = {"source_sha256": digest.hexdigest(), "method_version": METHOD_VERSION,
                "format": extension, "ocr": runtime, "parser_signature": representation_signature()}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if cache_root is None:
        from app.repository.settings import STORAGE_ROOT
        cache_root = Path(STORAGE_ROOT) / "representations"
    cache_path = Path(cache_root) / key[:2] / f"{key}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (cached.get("source_sha256") == digest.hexdigest() and cached.get("input_fingerprint") == key
                    and cached.get("status") == "text_ready"):
                return cached
        except (ValueError, OSError):
            # An interrupted or corrupt local cache is recoverable from original bytes.
            pass
    units, warnings, coverage = [], [], {}
    try:
        if extension == ".pdf":
            units, warnings, coverage = _parse_pdf(path, runtime, Path(cache_root) / "pages" / key)
        elif extension == ".docx":
            with ZipFile(path) as archive:
                info = archive.getinfo("word/document.xml")
                if info.file_size > 20 * 1024 * 1024:
                    raise ValueError("DOCX XML exceeds the 20 MiB parsing budget.")
            # Reuse the established exact XML/table locators used in graph evidence.
            from app.repository.knowledge_pipeline.adapters import _docx_parts
            parts, warnings = _docx_parts(path.read_bytes())
            for index, (locator, text) in enumerate(parts, 1):
                units.append(_unit(f"paragraph:{index}", index, text, "docx_xml", source_locator=locator))
        elif extension in {".txt", ".md", ".json", ".csv", ".tsv", ".html", ".htm"}:
            raw = path.read_bytes()
            if b"\x00" in raw or raw.startswith((b"PK\x03\x04", b"%PDF")):
                raise ValueError("Binary signature does not match a supported text document.")
            from app.repository.knowledge_pipeline.adapters import _json_parts, _text_parts, _utf8
            parts = _json_parts(raw) if extension == ".json" else _text_parts(_utf8(raw, "Document"))
            for index, (locator, text) in enumerate(parts, 1):
                units.append(_unit(f"chunk:{index}", index, text, "json_leaf" if extension == ".json" else "text_utf8", source_locator=locator))
        else:
            warnings.append(f"Unsupported binary or document format: {extension or '(none)'}. No text decoding attempted.")
    except Exception as exc:
        warnings.append(f"Document transcription failed: {type(exc).__name__}: {str(exc)[:200]}")
        if extension == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(path)
                unreadable = []
                for index, page in enumerate(reader.pages[:MAX_PAGES], 1):
                    text = _clean(page.extract_text() or "")
                    if text:
                        units.append(_unit(f"page:{index}", index, text, "native_pdf_pypdf"))
                    else:
                        unreadable.append(f"page:{index}")
                coverage = {"total_units": len(reader.pages), "readable_units": len(units),
                            "ocr_attempted": 0, "unreadable_locators": unreadable,
                            "omitted_units": max(0, len(reader.pages) - MAX_PAGES)}
                warnings.append("Used pypdf native-text fallback; page OCR was unavailable.")
            except Exception as fallback:
                warnings.append(f"pypdf fallback failed: {type(fallback).__name__}: {str(fallback)[:160]}")
    if not coverage:
        coverage = {"total_units": len(units), "readable_units": len(units), "ocr_attempted": 0,
                    "unreadable_locators": [], "omitted_units": 0}
    retryable_ocr = any(token in warning for warning in warnings
                        for token in ("OCR failed or timed out", "OCR unavailable", "OCR language data missing"))
    status = "unreadable_or_empty" if not units else (
        "partial_text" if coverage["unreadable_locators"] or coverage["omitted_units"] or retryable_ocr else "text_ready")
    result = {**identity, "input_fingerprint": key, "status": status, "units": units,
              "warnings": list(dict.fromkeys(warnings)), "coverage": coverage}
    result["representation_id"] = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    # Immutable versions retain every attempt. The input pointer may advance after
    # a failed OCR page recovers; successful pages are reused independently.
    _write_json(Path(cache_root) / "artifacts" / f"{result['representation_id']}.json", result, immutable=True)
    _write_json(cache_path, result)
    return result
