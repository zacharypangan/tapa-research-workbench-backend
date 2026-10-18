"""Content/model/context-addressed machine descriptions on existing storage."""

import hashlib
import json
from pathlib import Path
import uuid

PROMPT_VERSION = "observable-image-labels-v3"
_HASH_CACHE = {}


def description_input(image_path: str, context: str, model: str) -> dict:
    path = Path(image_path)
    stat = path.stat()
    stamp = (str(path), stat.st_size, stat.st_mtime_ns)
    image_hash = _HASH_CACHE.get(stamp)
    if image_hash is None:
        image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(_HASH_CACHE) >= 1024:
            _HASH_CACHE.clear()
        _HASH_CACHE[stamp] = image_hash
    inputs = {"image_sha256": image_hash, "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
              "vision_model": model, "prompt_version": PROMPT_VERSION}
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return {"key": key, **inputs, "context_char_count": len(context), "evidence_origin": "machine_visual_description"}


def record_path(storage_root, key):
    return Path(storage_root) / "derived" / "image_descriptions" / key[:2] / (key + ".json")


def read_record(storage_root, key):
    try:
        value = json.loads(record_path(storage_root, key).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (ValueError, OSError):
        return {}


def save_record(storage_root, record):
    path = record_path(storage_root, record["key"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
