"""Prepare inspectable image text without requiring an embedding provider.

The existing model/context-addressed description cache remains authoritative.
OCR and generated descriptions stay separate; vectors are a subsequent stage.
"""

import asyncio
import hashlib
import json
from pathlib import Path
import uuid

from . import image_descriptions


def _read_ocr(image_path, language, storage_root=None):
    from PIL import Image
    from .document_representation import ocr_image, ocr_runtime, representation_signature
    runtime = ocr_runtime(language)
    if not runtime["selected"]:
        return {"text": "", "state": "unavailable", "error_category": "ocr_not_available", "retryable": False}
    # Remember a successful empty transcription separately from a timeout. This
    # prevents every job retry from OCRing the same text-free figures again.
    inputs = {"image_sha256": image_descriptions.description_input(image_path, "", "")["image_sha256"],
              "runtime": runtime, "method": representation_signature(), "version": "image-ocr-empty/1"}
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    path = Path(storage_root) / "derived" / "image_ocr" / key[:2] / (key + ".json") if storage_root else None
    if path and path.is_file():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("inputs") == inputs and record.get("state") == "no_text":
                return {"text": "", "state": "no_text", "cache_reused": True, "language": "+".join(runtime["selected"])}
        except (OSError, ValueError):
            pass
    with Image.open(image_path) as image:
        result = ocr_image(image, runtime)
    failed = any(warning.startswith("OCR failed or timed out:") for warning in result.get("warnings", []))
    if path and not failed and not result["text"] and image_descriptions.description_input(image_path, "", "")["image_sha256"] == inputs["image_sha256"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps({"inputs": inputs, "state": "no_text"}), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"text": result["text"], "state": "failed" if failed else "readable" if result["text"] else "no_text",
            "confidence": result.get("confidence"), "language": result.get("language"),
            **({"error_category": "ocr_failed_or_timed_out", "retryable": True} if failed else {})}


def _description_current(repo, row, context):
    try:
        inputs = image_descriptions.description_input(row["image_path"], context, repo.OLLAMA_VISION_MODEL)
    except OSError:
        return False
    record = image_descriptions.read_record(repo.STORAGE_ROOT, inputs["key"])
    return bool(record.get("status") == "complete" and record.get("normalized_caption")
                and record["normalized_caption"] == (row["visual_caption"] or ""))


async def prepare_images(con, material_id, limit=200, skip_ids=None, progress_callback=None):
    from app.api import repository as repo

    if not 1 <= limit <= 200:
        raise ValueError("Image preparation batch limit must be between 1 and 200")
    material = repo.ensure_material(con, material_id)
    candidates = []
    for row in repo._image_index_rows(con, material_id):
        if row["id"] in (skip_ids or ()):
            continue
        context = repo.get_image_context_text(con, row)
        if not _description_current(repo, row, context) or not (row["ocr_text"] or "").strip():
            candidates.append((row, context))
    counters = dict(processed_image_count=0, captioned_count=0, ocr_count=0,
                    skipped_count=0, failed_count=0, caption_failed_count=0)
    outcomes = []
    for row, context in candidates[:limit]:
        counters["processed_image_count"] += 1
        try:
            if not Path(row["image_path"]).is_file():
                raise FileNotFoundError()
            if not await asyncio.to_thread(repo.is_informative_image, row["image_path"]):
                counters["skipped_count"] += 1
                outcomes.append({"id": row["id"], "status": "skipped", "reason": "Uninformative image"})
                continue
            old_ocr, old_caption = row["ocr_text"] or "", row["visual_caption"] or ""
            inputs = image_descriptions.description_input(row["image_path"], context, repo.OLLAMA_VISION_MODEL)
            ocr_result = {"text": old_ocr, "state": "existing"} if old_ocr else await asyncio.to_thread(_read_ocr, row["image_path"], material["language"], repo.STORAGE_ROOT)
            ocr = ocr_result["text"]
            ocr_detail = {key: value for key, value in ocr_result.items() if key != "text"}
            if image_descriptions.description_input(row["image_path"], context, repo.OLLAMA_VISION_MODEL)["key"] != inputs["key"]:
                outcomes.append({"id": row["id"], "status": "failed", "error_category": "source_changed", "retryable": True})
                counters["failed_count"] += 1
                continue
            # Persist readable OCR before making any network request. Failed vision
            # or embedding services must not discard useful local transcription.
            updated = con.execute("""UPDATE image_evidence SET ocr_text=?
                WHERE id=? AND image_path=? AND COALESCE(ocr_text,'')=? AND COALESCE(visual_caption,'')=?""",
                (ocr, row["id"], row["image_path"], old_ocr, old_caption))
            con.commit()
            if not updated.rowcount:
                outcomes.append({"id": row["id"], "status": "failed", "error_category": "source_changed", "retryable": True})
                counters["failed_count"] += 1
                continue
            counters["ocr_count"] += bool(ocr and not old_ocr)
            caption_failure = None
            try:
                caption = (await repo.request_image_caption(row["image_path"], context) or "").strip()
                if not caption:
                    caption_failure = repo._caption_failure(inputs)
            except Exception as exc:
                caption, caption_failure = "", repo._index_failure(exc)
            if caption:
                current = next(iter(repo._image_index_rows(con, material_id, row["id"])), None)
                if (current is None or repo.get_image_context_text(con, current) != context
                        or current["image_path"] != row["image_path"]
                        or image_descriptions.description_input(current["image_path"], context, repo.OLLAMA_VISION_MODEL)["key"] != inputs["key"]):
                    caption_failure = {"error_category": "source_changed", "retryable": True}
                else:
                    updated = con.execute("""UPDATE image_evidence SET visual_caption=?
                        WHERE id=? AND COALESCE(ocr_text,'')=? AND COALESCE(visual_caption,'')=?""",
                        (caption, row["id"], ocr, old_caption))
                    con.commit()
                    if not updated.rowcount:
                        caption_failure = {"error_category": "source_changed", "retryable": True}
            if caption_failure:
                counters["failed_count"] += 1
                counters["caption_failed_count"] += 1
                outcomes.append({"id": row["id"], "status": "failed", **caption_failure,
                                 "ocr_searchable": bool(ocr), "ocr": ocr_detail})
            elif ocr_result.get("error_category"):
                counters["captioned_count"] += 1
                counters["failed_count"] += 1
                outcomes.append({"id": row["id"], "status": "failed", "ocr_searchable": False, "ocr": ocr_detail,
                                 "error_category": ocr_result["error_category"], "retryable": ocr_result.get("retryable", False)})
            else:
                counters["captioned_count"] += 1
                outcomes.append({"id": row["id"], "status": "described", "ocr_searchable": bool(ocr),
                                 "ocr": ocr_detail})
        except Exception as exc:
            counters["failed_count"] += 1
            outcomes.append({"id": row["id"], "status": "failed", **repo._index_failure(exc)})
        finally:
            if progress_callback is not None:
                await progress_callback({**counters, "last_outcome": outcomes[-1],
                                         "pending_count": len(candidates) - len(outcomes)})
    return {**counters, "outcomes": outcomes, "pending_count": max(0, len(candidates) - limit),
            "vision_model": repo.OLLAMA_VISION_MODEL, "embedding_required": False,
            "evidence_origin": "machine_visual_description", "vision_diagnostics": repo.image_description_diagnostics()}
