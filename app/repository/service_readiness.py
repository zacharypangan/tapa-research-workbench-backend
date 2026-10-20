"""Capability reporting separates provider health from verified task execution.

These are operational checks, not evidence of translation or graph quality. No
provider URLs, credentials, or new inference requests are added by this mapping.
"""
from __future__ import annotations

import asyncio
import threading
import time


OCR_LANGUAGES = ("eng", "chi_sim", "chi_tra", "jpn")
_OCR_RUNTIME_CACHE: dict = {}
_OCR_RUNTIME_LOCK = threading.Lock()


def _cached_ocr_runtime() -> dict:
    # Worker threads share this cache. Tesseract discovery never blocks an API
    # event loop or launches one discovery subprocess per polling browser.
    with _OCR_RUNTIME_LOCK:
        cached = _OCR_RUNTIME_CACHE.get("snapshot")
        if cached and time.monotonic() - cached[0] < 60:
            return cached[1]
        try:
            from app.repository.document_representation import ocr_runtime
            result = ocr_runtime(list(OCR_LANGUAGES))
        except Exception as exc:
            result = {"engine_version": "unavailable", "installed": [],
                      "requested": list(OCR_LANGUAGES),
                      "warnings": [f"OCR runtime check failed: {type(exc).__name__}."]}
        _OCR_RUNTIME_CACHE["snapshot"] = (time.monotonic(), result)
        return result


async def local_ocr_runtime() -> dict:
    return await asyncio.to_thread(_cached_ocr_runtime)


def ocr_capability(runtime: dict) -> dict:
    installed = sorted(set(runtime.get("installed", [])))
    requested = list(OCR_LANGUAGES)
    missing = sorted(set(requested) - set(installed))
    version = runtime.get("engine_version", "unavailable")
    available = bool(version and version != "unavailable")
    ready = available and not missing
    state = "ready" if ready else "partial" if available else "unavailable"
    return {"configured": available, "ready": ready, "state": state,
            "provider": "local", "verification": "runtime_check_only",
            "engine_version": version, "requested_languages": requested,
            "installed_languages": installed, "missing_languages": missing,
            "detail": ((f"Tesseract {version} is installed. "
                        + (f"Missing requested language packs: {', '.join(missing)}. Installed languages remain usable. "
                           if missing else "English, Simplified Chinese, Traditional Chinese and Japanese language packs are installed. "))
                       if available else "Tesseract OCR is unavailable. Native PDF text extraction can still work. ")
                      + "This checks the runtime and language data; no document page or transcription quality was tested."}


def service_capabilities(*, generation_configured: bool, generation_ready: bool,
                         generation_detail: str, generation_provider: str,
                         generation_model: str, embedding_configured: bool,
                         embedding_ready: bool, embedding_detail: str,
                         embedding_provider: str, embedding_model: str,
                         vision_configured: bool, vision_model: str,
                         vision_failure: dict | None = None,
                         ocr_runtime: dict | None = None) -> dict:
    def probed(configured, ready, provider, model, detail):
        return {"configured": configured, "ready": ready,
                "state": "ready" if ready else "unavailable" if configured else "not_configured",
                "provider": provider, "model": model, "detail": detail,
                "verification": "provider_probe"}

    generation = probed(generation_configured, generation_ready, generation_provider,
                        generation_model, generation_detail)
    embeddings = probed(embedding_configured, embedding_ready, embedding_provider,
                        embedding_model, embedding_detail)
    services = {"generation": generation, "embeddings": embeddings}
    for key, task in (("translation", "Document translation"),
                      ("graph_extraction", "Evidence-grounded graph extraction")):
        services[key] = {**generation, "ready": None if generation_ready else False,
                         "state": "unverified" if generation_ready else generation["state"],
                         "depends_on": ["generation"], "dependency_ready": generation_ready,
                         "verification": "generation_probe_only",
                         "detail": (f"{task} is implemented. The text model responds; an actual task and its output validation have not been tested by this health check."
                                    if generation_ready else f"{task} requires the text model. {generation_detail}")}
    services["image_description"] = {
        "configured": vision_configured,
        "ready": False if vision_failure or not vision_configured else None,
        "state": "blocked" if vision_failure else "unverified" if vision_configured else "not_configured",
        "provider": generation_provider, "model": vision_model,
        "verification": "cached_failure" if vision_failure else "configuration_only",
        "detail": (f"Recent image description requests failed: {vision_failure.get('error_category', 'provider failure')}"
                   + (f" (HTTP {vision_failure['http_status']})." if vision_failure.get("http_status") else ".")
                   if vision_failure else
                   "A vision endpoint is configured. This health check has not tested image inference."
                   if vision_configured else "A vision model and reachable provider configuration are required."),
    }
    services["keyword_search"] = {
        "configured": True, "ready": True, "state": "ready", "provider": "local",
        "verification": "database_initialized",
        "detail": "Source-text and multilingual keyword search use the local database without a model provider.",
    }
    if ocr_runtime is not None:
        services["ocr"] = ocr_capability(ocr_runtime)
    return services
