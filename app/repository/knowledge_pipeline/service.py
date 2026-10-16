"""Checkpointed document → concepts → seeded web evidence → reviewed graph service."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import ipaddress
import math
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from .store import RunStore, digest, file_hash, now, read_json, write_json


def default_config() -> dict:
    return {"ontology": read_json(Path(__file__).with_name("default_ontology.json")),
            "processing": {"mode": "manual", "batch_chars": 12000},
            "augmentation": {"enabled": True, "max_pages": 5, "delay": 1.0, "timeout": 20.0},
            "research_topic": "tapa barkcloth manufacture archaeology"}


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict) or set(config) - {"ontology", "research_topic", "processing", "augmentation"}:
        raise ValueError("Unsupported pipeline configuration")
    result = default_config()
    for key in ("ontology", "research_topic"):
        if key in config:
            result[key] = config[key]
    for key in ("processing", "augmentation"):
        if key in config and not isinstance(config[key], dict):
            raise ValueError(f"{key} must be an object")
        result[key].update(config.get(key, {}))
    ontology = result["ontology"]
    if not isinstance(ontology, dict) or not isinstance(ontology.get("entity_types"), dict) or not isinstance(ontology.get("predicates"), dict) or not ontology["entity_types"] or not ontology["predicates"]:
        raise ValueError("Ontology requires entity_types and predicates")
    if not isinstance(ontology.get("version"), str) or not ontology["version"].strip():
        raise ValueError("Ontology requires an explicit version")
    if max(len(ontology["entity_types"]), len(ontology["predicates"])) > 256:
        raise ValueError("Ontology exceeds 256 types or predicates")
    if not isinstance(result["processing"]["mode"], str) or result["processing"]["mode"] not in {"manual", "ollama"}:
        raise ValueError("Processing mode must be manual or ollama")
    if type(result["processing"]["batch_chars"]) is not int or not 1000 <= result["processing"]["batch_chars"] <= 24000:
        raise ValueError("batch_chars must be between 1000 and 24000")
    augmentation = result["augmentation"]
    if type(augmentation["enabled"]) is not bool:
        raise ValueError("augmentation.enabled must be a boolean")
    if type(augmentation["max_pages"]) is not int or not 1 <= augmentation["max_pages"] <= 25:
        raise ValueError("Collection requires 1–25 pages")
    if type(augmentation["delay"]) not in {int, float} or not math.isfinite(augmentation["delay"]) or not 0.5 <= augmentation["delay"] <= 30:
        raise ValueError("Collection requires 1–25 pages and a delay of 0.5–30 seconds")
    if type(augmentation["timeout"]) not in {int, float} or not math.isfinite(augmentation["timeout"]) or not 1 <= augmentation["timeout"] <= 120:
        raise ValueError("Collection timeout must be 1–120 seconds")
    # Public API configuration cannot select browser binaries, credentials or arbitrary code.
    permitted = {"enabled", "max_pages", "delay", "timeout"}
    if set(augmentation) - permitted or set(result["processing"]) - {"mode", "batch_chars"}:
        raise ValueError("Unsupported pipeline option")
    if not isinstance(result["research_topic"], str) or len(result["research_topic"]) > 500:
        raise ValueError("research_topic must contain at most 500 characters")
    from .processing import extraction_schema
    extraction_schema(ontology)
    return result


def batches_for(units: list[dict], limit: int) -> list[dict]:
    """Split long pages with explicit parent offsets; never silently truncate a source."""
    groups = {}
    for unit in units:
        text = unit["text"]
        pieces = []
        for offset in range(0, len(text), limit):
            piece = copy.deepcopy(unit)
            piece["text"] = text[offset:offset + limit]
            piece["text_sha256"] = hashlib.sha256(piece["text"].encode()).hexdigest()
            if len(text) > limit:
                piece["unit_id"] = "UNIT_" + digest([unit["unit_id"], offset, offset + len(piece["text"])])[:24]
                piece["locator"] = {"parent_locator": unit["locator"], "parent_unit_id": unit["unit_id"],
                                    "start_char": offset, "end_char": offset + len(piece["text"])}
            pieces.append(piece)
        groups.setdefault(unit["source_id"], []).extend(pieces)
    batches = []
    for source_id, source_units in groups.items():
        current, size = [], 0
        for unit in source_units:
            if current and size + len(unit["text"]) > limit:
                batches.append({"source_id": source_id, "units": current})
                current, size = [], 0
            current.append(unit)
            size += len(unit["text"])
        if current:
            batches.append({"source_id": source_id, "units": current})
    for batch in batches:
        batch["batch_id"] = "BATCH_" + digest(batch)[:24]
    return batches


def merge_outputs(outputs: list[dict]) -> dict:
    result = {}
    for field, key in (("entities", "entity_id"), ("assertions", "assertion_id"), ("evidence", "evidence_id")):
        records = {}
        for output in outputs:
            for record in output.get(field, []):
                identifier = record[key]
                if identifier in records and records[identifier] != record:
                    raise ValueError(f"Conflicting version for {identifier}")
                records[identifier] = record
        result[field] = list(records.values())
    result["rejections"] = [item for output in outputs for item in output.get("rejections", [])]
    return result


def verify_implementation(inputs: dict) -> None:
    for filename, expected in inputs.get("implementation_sha256", {}).items():
        implementation = Path(__file__).with_name(filename)
        if not implementation.is_file() or file_hash(implementation) != expected:
            raise ValueError("Pipeline implementation changed; use the pinned code version or create a new run")


async def blocking_stage(function, *args):
    """Retain the run lock until a bounded file-writing thread actually stops."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # to_thread cancellation cannot stop a collector/parser. Releasing the
        # lock early would let a retry write into its still-active destination.
        try:
            await task
        except Exception:
            pass
        raise


class ResearchPipeline:
    def __init__(self, store: RunStore, provider=None, collector=None):
        self.store, self.provider, self.collector = store, provider, collector

    async def _process(self, path: Path, name: str, units: list[dict], config: dict) -> dict | None:
        from .processing import process_response

        complete = path / f"{name}.json"
        if complete.exists():
            return read_json(complete)
        batches = batches_for(units, int(config["processing"]["batch_chars"]))
        write_json(path / f"{name}_batches.json", batches)
        self.store.state(path, "running", name)
        responses = read_json(path / "responses.json", {})
        missing, outputs = [], []
        for batch in batches:
            identifier = batch["batch_id"]
            checkpoint = path / "processed" / f"{identifier}.json"
            if checkpoint.exists():
                outputs.append(read_json(checkpoint))
                continue
            metadata = {"method": "submitted_annotation", "ontology_sha256": digest(config["ontology"])}
            if identifier in responses:
                response = responses[identifier]
                output = process_response(response, batch["units"], config["ontology"])
                metadata["response_sha256"] = digest(response)
                write_json(path / "provider_responses" / f"{identifier}.json", response)
            elif config["processing"]["mode"] == "ollama" and self.provider:
                output = await self.provider(batch["units"], config["ontology"])
                metadata = output.pop("provider_metadata", {})
                response = output.pop("provider_response", None)
                if response is not None:
                    write_json(path / "provider_responses" / f"{identifier}.json", response)
            else:
                missing.append(identifier)
                continue
            output["processing_metadata"] = metadata
            write_json(checkpoint, output)
            outputs.append(output)
        if missing:
            self.store.state(path, "awaiting_annotations", name, missing_batch_ids=missing,
                             detail="Submit source-grounded annotations or configure the existing Ollama runtime")
            return None
        result = merge_outputs(outputs)
        result["processing_runs"] = [{"batch_id": batch["batch_id"],
                                      **read_json(path / "processed" / f"{batch['batch_id']}.json")["processing_metadata"]} for batch in batches]
        write_json(complete, result)
        return result

    async def run(self, run_id: str) -> dict:
        path = self.store.path(run_id)
        with self.store.lock(path):
            try:
                await self._advance(path)
            except asyncio.CancelledError:
                prior = read_json(path / "state.json")
                self.store.state(path, "failed", prior.get("stage", "unknown"), error_type="CancelledError",
                                 detail="Execution interrupted; completed checkpoints remain resumable")
                raise
            except Exception as exc:
                prior = read_json(path / "state.json")
                self.store.state(path, "failed", prior.get("stage", "unknown"),
                                 error_type=type(exc).__name__, detail="Stage failed; preserved checkpoints can be resumed")
                raise
        return self.store.describe(run_id)

    async def _advance(self, path: Path):
        from .adapters import collect_frontier, ingest_document
        from .processing import build_alignment_candidates

        inputs = read_json(path / "inputs.json")
        if read_json(path / "state.json")["status"] == "released":
            return
        verify_implementation(inputs)
        config = inputs["config"]
        self.store.state(path, "running", "ingest")
        ingest_path = path / "ingest.json"
        if not ingest_path.exists():
            sources, units = [], []
            for document in inputs["documents"]:
                raw = path / document["input_path"]
                if file_hash(raw) != document["sha256"]:
                    raise ValueError("Immutable document snapshot was modified")
                destination = path / "documents" / document["source_id"]
                # Interrupted adapters must not expose a partial checkpoint as complete.
                if destination.exists():
                    import shutil
                    shutil.rmtree(destination)
                result = await blocking_stage(ingest_document, raw, document["source_id"], document["title"], destination)
                result["source"]["source_locator"] = document["original_locator"]
                sources.append(result["source"])
                units.extend(result["units"])
            write_json(ingest_path, {"sources": sources, "units": units})
        ingest = read_json(ingest_path)
        if not ingest["units"]:
            self.store.state(path, "awaiting_readable_documents", "ingest", detail="No readable text. Supply OCR or a new document version; empty text is not evidence of absence")
            return
        self.store.state(path, "running", "primary")
        primary = await self._process(path, "primary", ingest["units"], config)
        if primary is None:
            return
        by_id = {entity["entity_id"]: entity for entity in primary["entities"]}
        sources_by_id = {source["source_id"]: source for source in ingest["sources"]}
        seeds = []
        for entity in primary["entities"]:
            relations = [assertion for assertion in primary["assertions"] if assertion["subject_id"] == entity["entity_id"]]
            label = entity.get("label_original", entity.get("label", entity["entity_id"]))
            seeds.append({"seed_id": entity["entity_id"], "entity_id": entity["entity_id"], "source_id": entity["source_id"],
                          "label": label, "entity_type": entity["entity_type"], "evidence_ids": entity["evidence_ids"],
                          "reported_relationship_ids": [relation["assertion_id"] for relation in relations],
                          "question": f"What independently documented information extends the source's account of {label}?",
                          "queries": [f'"{label}" {config["research_topic"]}',
                                      f'"{label}" ' + " ".join(by_id.get(relation.get("object_id"), {}).get("label_original", "") for relation in relations[:3])],
                          "discovery_status": "awaiting_seed_bound_urls"})
            sources_by_id[entity["source_id"]].setdefault("seeds", []).append({"seed_id": entity["entity_id"], "entity_id": entity["entity_id"]})
        write_json(path / "seeds.json", seeds)
        if config["augmentation"]["enabled"] and seeds and not (path / "frontier.json").exists():
            self.store.state(path, "awaiting_frontier", "augmentation", seed_count=len(seeds),
                             detail="Register discovery URLs against document-derived seeds before collection")
            return
        frontier = read_json(path / "frontier.json", []) if seeds else []
        self.store.state(path, "running", "augmentation")
        web_path = path / "web.json"
        if not web_path.exists():
            if frontier:
                options = {key: value for key, value in config["augmentation"].items() if key != "enabled"}
                destination = path / "collections" / "round-001"
                if destination.exists():
                    # Preserve interrupted attempts separately, retry into a fresh destination.
                    destination.rename(destination.with_name("interrupted-" + now().replace(":", "-")))
                collected = await blocking_stage(self.collector or collect_frontier, frontier, destination, options)
            else:
                collected = {"sources": [], "units": [], "attempts": []}
            write_json(web_path, collected)
        web = read_json(web_path)
        augmented = await self._process(path, "augmented", web["units"], config)
        if augmented is None:
            return
        all_sources = list(sources_by_id.values()) + web["sources"]
        all_evidence = primary["evidence"] + augmented["evidence"]
        alignments = build_alignment_candidates(primary["entities"], augmented["entities"], all_evidence, config["ontology"])
        web_sources = {source["source_id"]: source for source in web["sources"]}
        web_entities = {entity["entity_id"]: entity for entity in augmented["entities"]}
        linked = []
        for alignment in alignments:
            seed_id = alignment["subject_id"]
            web_source = web_sources[web_entities[alignment["object_id"]]["source_id"]]
            if seed_id not in web_source.get("seed_ids", []):
                continue
            alignment["seed_ids"] = [seed_id]
            alignment.setdefault("scope", {})["seed_ids"] = [seed_id]
            linked.append(alignment)
        graph_input = {"sources": all_sources, "entities": primary["entities"] + augmented["entities"],
                       "evidence": all_evidence, "assertions": primary["assertions"] + augmented["assertions"], "alignments": linked}
        write_json(path / "graph_input.json", graph_input)
        primary_keys = {(e["entity_type"], str(e.get("label_original", "")).casefold()) for e in primary["entities"]}
        novel = [e for e in augmented["entities"] if (e["entity_type"], str(e.get("label_original", "")).casefold()) not in primary_keys]
        successful = len(web["sources"])
        write_json(path / "augmentation_metrics.json", {"successful_snapshots": successful,
                   "novel_source_local_candidates": len(novel), "average_novel_candidates_per_snapshot": len(novel) / successful if successful else None,
                   "seed_bound_alignment_candidates": len(linked), "collection_attempts": web.get("attempts", []),
                   "interpretation": "Candidate yield, not accepted scientific facts or a demonstrated saturation threshold"})
        self.store.state(path, "awaiting_review", "release", entity_count=len(graph_input["entities"]),
                         assertion_count=len(graph_input["assertions"]), alignment_count=len(linked),
                         rejection_count=len(primary["rejections"]) + len(augmented["rejections"]))

    def submit_responses(self, run_id: str, responses: dict) -> dict:
        path = self.store.path(run_id)
        with self.store.lock(path):
            state = read_json(path / "state.json")
            if state["status"] != "awaiting_annotations" and not (state["status"] == "failed" and state["stage"] in {"primary", "augmented"}):
                raise ValueError("Run is not awaiting annotations")
            known = {batch["batch_id"] for name in ("primary", "augmented")
                     for batch in read_json(path / f"{name}_batches.json", [])}
            existing = read_json(path / "responses.json", {})
            for key, value in responses.items():
                if key not in known:
                    raise ValueError("Unknown annotation batch")
                if key in existing and existing[key] != value:
                    raise ValueError("Annotation is immutable; create a new run for corrected source processing")
                if (path / "processed" / f"{key}.json").exists() and key not in existing:
                    raise ValueError("Batch already processed by provider")
            write_json(path / "responses.json", {**existing, **responses})
        return self.store.describe(run_id)

    def skip_augmentation(self, run_id: str, reason: str, actor: str) -> dict:
        """Record a researcher choice to finish with document evidence only."""
        path = self.store.path(run_id)
        with self.store.lock(path):
            if read_json(path / "state.json")["status"] != "awaiting_frontier":
                raise ValueError("Web enrichment can be skipped only when the run is awaiting web sources")
            if not isinstance(reason, str) or not 3 <= len(reason.strip()) <= 2000:
                raise ValueError("Skipping web enrichment requires a reason of 3–2000 characters")
            write_json(path / "augmentation_policy_decision.json", {
                "action": "skip_by_user", "actor_id": actor, "at": now(), "reason": reason.strip(),
                "previous_frontier": read_json(path / "frontier.json", []),
            })
            write_json(path / "frontier.json", [])
        return self.store.describe(run_id)

    def submit_frontier(self, run_id: str, frontier: list[dict], actor: str) -> dict:
        path = self.store.path(run_id)
        with self.store.lock(path):
            if read_json(path / "state.json")["status"] != "awaiting_frontier":
                raise ValueError("Run is not awaiting a collection frontier")
            if not 1 <= len(frontier) <= 25:
                raise ValueError("Register 1–25 URLs")
            seed_rows = {seed["seed_id"]: seed for seed in read_json(path / "seeds.json")}
            seeds = set(seed_rows)
            budget = read_json(path / "inputs.json")["config"]["augmentation"]["max_pages"]
            if len({item["url"] for item in frontier}) > budget:
                raise ValueError("Frontier exceeds the pinned collection page budget")
            for item in frontier:
                url = urlsplit(item["url"])
                if url.scheme not in {"https", "http"} or not url.hostname or url.username or url.password or url.fragment:
                    raise ValueError("Use a public HTTP(S) URL without credentials or fragments")
                hostname = url.hostname.casefold().rstrip(".")
                if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
                    raise ValueError("Collection requires a public host")
                try:
                    address = ipaddress.ip_address(hostname)
                except ValueError:
                    address = None
                if address is not None and not address.is_global:
                    raise ValueError("Collection requires a public address")
                if not item.get("seed_ids") or not set(item["seed_ids"]).issubset(seeds):
                    raise ValueError("Every URL must resolve to document-derived seed IDs")
                if not item.get("question", "").strip():
                    item["question"] = " ".join(seed_rows[seed]["question"] for seed in item["seed_ids"])
                item["registered_by"], item["registered_at"] = actor, now()
                item.setdefault("discovery_method", "registered_url")
            write_json(path / "frontier.json", frontier)
        return self.store.describe(run_id)

    def review(self, run_id: str, decisions: list[dict], actor: str) -> dict:
        path = self.store.path(run_id)
        with self.store.lock(path):
            if read_json(path / "state.json")["status"] != "awaiting_review":
                raise ValueError("Run is not awaiting review")
            graph = read_json(path / "graph_input.json")
            candidates = {row[key] for field, key in (("entities", "entity_id"), ("assertions", "assertion_id"), ("alignments", "alignment_id")) for row in graph[field]}
            journal = read_json(path / "decisions.json", [])
            for decision in decisions:
                if decision["candidate_id"] not in candidates or decision["decision"] not in {"accept", "reject"} or not decision.get("rationale", "").strip():
                    raise ValueError("Review requires a known candidate, accept/reject and rationale")
                journal.append({**decision, "actor_id": actor, "at": now()})
            write_json(path / "decisions.json", journal)
        return self.store.describe(run_id)

    def release(self, run_id: str, actor: str, allow_pending: bool = False) -> dict:
        from .graph import build_release

        path = self.store.path(run_id)
        with self.store.lock(path):
            if read_json(path / "state.json")["status"] != "awaiting_review":
                raise ValueError("Run is not awaiting review")
            graph = read_json(path / "graph_input.json")
            inputs = read_json(path / "inputs.json")
            verify_implementation(inputs)
            release_root = path / "releases"
            release_root.mkdir(exist_ok=True)
            sequence = len(list(release_root.glob("release-*"))) + 1
            destination = release_root / f"release-{sequence:04d}"
            staging = release_root / (".stage-" + uuid.uuid4().hex)
            decisions = read_json(path / "decisions.json", [])
            summary = build_release(staging, inputs["config"]["ontology"], decisions=decisions, allow_pending=allow_pending, **graph)
            write_json(staging / "run_lineage.json", {"run_id": run_id, "fingerprint": inputs["fingerprint"],
                       "pipeline_version": inputs["pipeline_version"], "ontology_sha256": inputs["ontology_sha256"],
                       "released_by": actor, "at": now(), "preview": allow_pending,
                       "checkpoint_hashes": {file.name: file_hash(file) for file in path.glob("*.json")}})
            manifest = read_json(staging / "manifest.json")
            lineage_hash = file_hash(staging / "run_lineage.json")
            manifest["artifact_sha256"]["run_lineage.json"] = lineage_hash
            write_json(staging / "manifest.json", manifest)
            summary["artifact_sha256"].update({"run_lineage.json": lineage_hash, "manifest.json": file_hash(staging / "manifest.json")})
            summary["manifest_sha256"] = summary["artifact_sha256"]["manifest.json"]
            staging.rename(destination)
            self.store.state(path, "awaiting_review" if allow_pending else "released", "release",
                             release_path=destination.relative_to(path).as_posix(), release_summary=summary)
        return self.store.describe(run_id)
