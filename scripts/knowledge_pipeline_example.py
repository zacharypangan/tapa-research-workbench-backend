"""Runnable offline demonstration, using synthetic text and authored annotations.

Run from stling_backend with the existing environment:
  .venv/bin/python -m scripts.knowledge_pipeline_example --output ../tmp/kg-example

The injected Python fixture provider makes no Ollama/LLM request. The collector
copies a synthetic TXT fixture; its registered URL is never fetched. This creates
a pending preview for demonstrating software behavior, not research findings.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path

from app.repository.knowledge_pipeline.adapters import ingest_document
from app.repository.knowledge_pipeline.processing import process_response
from app.repository.knowledge_pipeline.service import ResearchPipeline, validate_config
from app.repository.knowledge_pipeline.store import RunStore, digest, read_json, write_json


def authored_response(units: list[dict]) -> dict:
    """Known annotations for these fixtures only; this is not an extractor."""
    evidence = [{"unit_id": units[0]["unit_id"], "quote": units[0]["text"]}]
    return {
        "entities": [
            {"local_id": "case", "entity_type": "ProductionCase", "label_original": "Example case", "evidence": evidence},
            {"local_id": "stage", "entity_type": "ActivityType", "label_original": "soaking", "evidence": evidence},
            {"local_id": "material", "entity_type": "MaterialKind", "label_original": "barkcloth", "evidence": evidence},
        ],
        "assertions": [{
            "local_id": "processing", "subject": "stage", "predicate": "processes_material", "object": "material",
            "scope": {"case_id": "case", "synthetic_fixture": True}, "polarity": "positive",
            "epistemic_kind": "source_report", "evidence": evidence,
            "rationale": "Manually authored annotation of an explicitly synthetic sentence; no real production practice is inferred.",
        }],
    }


async def demonstrate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    fixtures = output / "fixtures"
    fixtures.mkdir()
    primary_path = fixtures / "synthetic-primary.txt"
    primary_path.write_text("Synthetic primary document: In Example case, soaking processes barkcloth.\n", encoding="utf-8")
    web_path = fixtures / "synthetic-web.txt"
    web_path.write_text("Synthetic offline web fixture: In Example case, soaking processes barkcloth.\n", encoding="utf-8")

    async def fixture_provider(units, ontology):
        response = authored_response(units)
        return {
            **process_response(response, units, ontology),
            "provider_response": response,
            "provider_metadata": {
                "method": "manually_authored_synthetic_fixture", "provider": "offline_python_fixture",
                "response_sha256": digest(response), "ontology_sha256": digest(ontology),
                "network_calls": False, "model_inference": False,
            },
        }

    def fixture_collector(frontier, destination, options):
        # Use the real acquisition adapter; all data remain explicitly synthetic.
        ingested = ingest_document(web_path, "WEB_SYNTHETIC", "Synthetic offline web fixture; URL not fetched", destination, origin="web")
        source = copy.deepcopy(ingested["source"])
        units = copy.deepcopy(ingested["units"])
        seeds = sorted({seed for entry in frontier for seed in entry["seed_ids"]})
        for unit in units:
            unit["seed_ids"] = seeds
        # Register the derived seed-bound representation separately, preserving
        # the original adapter output and its original JSONL hash.
        representation = b"".join(
            (json.dumps(unit, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            for unit in units
        )
        representation_path = destination / "seed_bound_units.jsonl"
        with representation_path.open("xb") as stream:
            stream.write(representation)
        representation_path.chmod(0o400)
        source.update({
            "seed_ids": seeds, "source_kind": "synthetic_web_fixture",
            "source_locator": "synthetic:offline-web-fixture",
            "representation_sha256": hashlib.sha256(representation).hexdigest(),
            "representation_path": str(representation_path), "acquisition_method": "offline_synthetic_fixture_copy",
            "warnings": ["Synthetic fixture; no URL was fetched and no real web research was performed."],
        })
        write_json(destination / "seed_binding.json", {"source": source, "frontier": frontier,
                   "original_representation_sha256": ingested["source"]["representation_sha256"],
                   "network_calls": False})
        return {"sources": [source], "units": units,
                "attempts": [{"status": "offline_synthetic_fixture", "seed_ids": seeds, "network_calls": False}]}

    config = validate_config({
        # This selector invokes the injected fixture provider, not an Ollama call.
        "processing": {"mode": "ollama"}, "augmentation": {"enabled": True, "max_pages": 1},
        "research_topic": "explicitly synthetic offline software demonstration",
    })
    store = RunStore(output / "runs", "offline-synthetic-demonstration")
    pipeline = ResearchPipeline(store, provider=fixture_provider, collector=fixture_collector)
    created = store.create([{"path": primary_path, "title": "Synthetic primary document", "locator": "synthetic:primary-fixture"}],
                           config, "offline-example-author")
    run_id = created["run_id"]
    state = await pipeline.run(run_id)
    if state["status"] != "awaiting_frontier":
        raise RuntimeError("Synthetic primary processing did not reach seed registration")
    seeds = read_json(store.path(run_id) / "seeds.json")
    pipeline.submit_frontier(run_id, [{
        "url": "https://example.org/offline-synthetic-fixture-not-fetched",
        "seed_ids": [seed["seed_id"] for seed in seeds],
        "question": "Demonstrate source-bound alignment only.", "discovery_method": "offline_synthetic_fixture",
    }], "offline-example-author")
    state = await pipeline.run(run_id)
    if state["status"] != "awaiting_review":
        raise RuntimeError("Synthetic augmentation did not reach review")
    released = pipeline.release(run_id, "offline-example-author", allow_pending=True)
    graph_path = store.path(run_id) / released["release_path"] / "graph.json"
    summary = {
        "demonstration": "synthetic_offline_pending_preview", "run_id": run_id,
        "graph_path": str(graph_path), "counts": released["release_summary"]["counts"],
        "network_calls": False, "model_inference": False,
        "limits": "Synthetic software demonstration only; all scientific and translation review remain pending.",
    }
    write_json(output / "example_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path, help="Fresh output directory; existing directories are refused")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("--output must identify a fresh directory")
    print(json.dumps(asyncio.run(demonstrate(output)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
