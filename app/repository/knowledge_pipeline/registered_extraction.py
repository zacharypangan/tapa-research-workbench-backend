"""Immutable graph inputs from already registered extraction segments.

These are explicit derived representations, never relabeled original documents.
Metadata is retained as provenance and is not passed to concept extraction.
"""

from pathlib import Path
import platform

from .adapters import _fresh_directory, _json_bytes, _jsonl_bytes, _sha, _unit, _write
from .store import file_hash, read_json, write_json


def registered_documents(material_id: str, file_ids: list[str] | None = None) -> list[dict]:
    from app.api import repository

    root = Path(repository.FILES_ROOT).resolve()
    with repository.get_connection() as connection:
        material = dict(repository.ensure_material(connection, material_id))
        files = connection.execute("SELECT * FROM files WHERE material_id=? ORDER BY id", (material_id,)).fetchall()
        requested = set(file_ids) if file_ids is not None else {row["id"] for row in files}
        if not requested or not requested.issubset({row["id"] for row in files}):
            raise ValueError("Select registered files belonging to this document")
        if len(requested) > 20:
            raise ValueError("Select at most 20 registered files")
        extraction = connection.execute(
            "SELECT id, status, created_at, error_message FROM extraction_runs WHERE material_id=? ORDER BY created_at DESC LIMIT 1",
            (material_id,),
        ).fetchone()
        documents = []
        for row in files:
            if row["id"] not in requested:
                continue
            original = Path(row["stored_path"]).resolve()
            if not original.is_relative_to(root) or not original.is_file():
                raise ValueError("Original file must remain in managed file storage")
            segments = connection.execute(
                "SELECT id, source_kind, source_locator, page_ref, page_index, content_text, created_at "
                "FROM extracted_segments WHERE material_id=? AND file_id=? AND source_kind NOT GLOB 'historical_*' "
                "ORDER BY page_index, id", (material_id, row["id"]),
            ).fetchall()
            segments = [dict(segment) for segment in segments if segment["content_text"].strip()]
            if not segments:
                raise ValueError(f"No registered readable extraction for {row['original_filename']}; extract this file first")
            payload = {
                "representation_kind": "registered_extraction", "version": "registered-extraction/1",
                "material_id": material_id, "file_id": row["id"],
                "parent_file": {"locator": f"workbench:material/{material_id}/file/{row['id']}",
                    "original_filename": row["original_filename"], "sha256": file_hash(original),
                    "byte_count": original.stat().st_size},
                "extraction_run": {**dict(extraction), "scope": "latest_material_extraction_record"} if extraction else None,
                "association_basis": "Registered extraction file_id association; original text accuracy requires source review",
                "segments": segments,
            }
            data = _json_bytes(payload)
            if len(data) > 50 * 1024 * 1024:
                raise ValueError("Registered extraction exceeds the 50 MiB graph input limit")
            target = Path(repository.STORAGE_ROOT) / "graph_representations" / (_sha(data) + ".json")
            if not target.exists():
                write_json(target, payload)
            elif target.read_bytes() != data:
                # write_json uses pretty JSON; compare canonical content, not formatting.
                if _json_bytes(read_json(target)) != data:
                    raise ValueError("Registered extraction snapshot hash collision")
            documents.append({"path": target, "title": material["title"] if len(requested) == 1 else row["original_filename"],
                "filename": row["original_filename"] + ".registered-extraction.json",
                "locator": payload["parent_file"]["locator"], "representation_kind": "registered_extraction"})
    return documents


def ingest_registered_extraction(path: Path, source_id: str, title: str, output_dir: Path) -> dict:
    data = Path(path).read_bytes()
    payload = read_json(Path(path))
    if payload.get("representation_kind") != "registered_extraction" or payload.get("version") != "registered-extraction/1":
        raise ValueError("Unsupported registered extraction representation")
    raw_hash = _sha(data)
    version = source_id + ":" + raw_hash
    units = []
    for segment in payload["segments"]:
        text = segment["content_text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Registered extraction segment must contain readable text")
        locator = {"kind": "registered_extraction_segment", "material_id": payload["material_id"],
            "file_id": payload["file_id"], "segment_id": segment["id"],
            **{key: segment[key] for key in ("source_kind", "source_locator", "page_ref", "page_index")},
            "original_file_sha256": payload["parent_file"]["sha256"],
            "extraction_created_at": segment["created_at"],
            "text_basis": "exact registered extraction text; original transcription requires source review"}
        unit = _unit(source_id, version, locator, text)
        unit["representation_method"] = "registered_extraction_snapshot"
        units.append(unit)
    representation = _jsonl_bytes(units)
    directory = _fresh_directory(output_dir)
    snapshot = directory / "snapshot.registered-extraction.json"
    representation_path = directory / "units.jsonl"
    _write(snapshot, data)
    _write(representation_path, representation)
    source = {"source_id": source_id, "source_version_id": version, "title": title, "origin": "primary",
        "source_kind": "registered_extraction", "representation_kind": "registered_extraction",
        "source_locator": payload["parent_file"]["locator"], "raw_sha256": raw_hash, "snapshot_sha256": raw_hash,
        "representation_sha256": _sha(representation), "snapshot_path": str(snapshot),
        "representation_path": str(representation_path), "acquisition_method": "registered_extraction_snapshot",
        "extractor": "registered-extraction/1", "extractor_code_sha256": file_hash(Path(__file__)),
        "python_version": platform.python_version(), "status": "derived_text", "seed_ids": [],
        "review_status": "needs_review", "unreadable_locators": [], "unit_count": len(units),
        "derived_from": payload["parent_file"], "extraction_run": payload["extraction_run"],
        "warnings": ["Derived from registered extraction segments; original bytes were preserved. "
                     "Layout, table interpretation and transcription accuracy require source review."]}
    _write(directory / "source.json", _json_bytes(source))
    return {"source": source, "units": units}
