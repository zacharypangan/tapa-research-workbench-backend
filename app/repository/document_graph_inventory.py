"""Read-only per-file extraction identities for complete graph backfills."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.repository.knowledge_pipeline.store import file_hash


def extraction_inventory(material_id):
    from app.api import repository
    root = Path(repository.FILES_ROOT).resolve()
    with repository.get_connection() as connection:
        material = dict(repository.ensure_material(connection, material_id))
        files = connection.execute("SELECT * FROM files WHERE material_id=? ORDER BY id", (material_id,)).fetchall()
        extraction = connection.execute(
            "SELECT id, status, created_at, error_message FROM extraction_runs WHERE material_id=? ORDER BY created_at DESC LIMIT 1",
            (material_id,),
        ).fetchone()
        unassociated = connection.execute(
            "SELECT COUNT(*) FROM extracted_segments WHERE material_id=? AND file_id IS NULL "
            "AND source_kind NOT GLOB 'historical_*' AND length(trim(content_text))>0", (material_id,),
        ).fetchone()[0]
        result = []
        for row in files:
            item = {"file_id": row["id"], "original_filename": row["original_filename"],
                    "eligible": False, "segment_count": 0, "character_count": 0,
                    "snapshot_sha256": None, "original_sha256": None}
            original = Path(row["stored_path"]).resolve()
            if not original.is_relative_to(root) or not original.is_file() or original.is_symlink():
                item["reason"] = "Original file is unavailable in managed storage"
                result.append(item)
                continue
            segments = [dict(segment) for segment in connection.execute(
                "SELECT id, source_kind, source_locator, page_ref, page_index, content_text, created_at "
                "FROM extracted_segments WHERE material_id=? AND file_id=? AND source_kind NOT GLOB 'historical_*' "
                "ORDER BY page_index, id", (material_id, row["id"]),
            ).fetchall() if segment["content_text"].strip()]
            item.update(segment_count=len(segments), character_count=sum(len(s["content_text"]) for s in segments))
            if not segments:
                item["reason"] = "No registered readable extraction; extract this file first"
                result.append(item)
                continue
            try:
                sha = file_hash(original)
                payload = {"representation_kind": "registered_extraction", "version": "registered-extraction/1",
                           "material_id": material_id, "file_id": row["id"],
                           "parent_file": {"locator": f"workbench:material/{material_id}/file/{row['id']}",
                                           "original_filename": row["original_filename"], "sha256": sha,
                                           "byte_count": original.stat().st_size},
                           "extraction_run": {**dict(extraction), "scope": "latest_material_extraction_record"} if extraction else None,
                           "association_basis": "Registered extraction file_id association; original text accuracy requires source review",
                           "segments": segments}
                # Exactly the bytes registered_documents writes via write_json.
                encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
                item.update(original_sha256=sha, snapshot_sha256=hashlib.sha256(encoded).hexdigest())
                if len(encoded) > 50 * 1024 * 1024:
                    item["reason"] = "Registered extraction exceeds the 50 MiB graph input limit"
                else:
                    item["eligible"] = True
            except OSError:
                item["reason"] = "Original file could not be read; retry after storage is available"
            result.append(item)
    return {"material_id": material_id, "title": material.get("title"), "files": result,
            "unassociated_segment_count": unassociated,
            "extraction_status": extraction["status"] if extraction else "not_extracted",
            "interpretation": "File-associated original extraction coverage; link-only segments are reported separately"}
