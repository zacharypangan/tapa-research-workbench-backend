"""Local replay/worker interface; HTTP authorization remains in research_runs.py."""

import argparse
import asyncio
import json
from pathlib import Path

from .processing import call_ollama, extraction_schema
from .service import ResearchPipeline, default_config, validate_config
from .store import RunStore, read_json
from .runtime import collector_for_runtime, existing_collector_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Artifact storage root")
    parser.add_argument("--organization", required=True, help="Local storage namespace; CLI is operator-only")
    parser.add_argument("--actor", default="local-operator")
    parser.add_argument("--collector-python", type=Path, help="Operator-selected existing collection interpreter")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("documents", nargs="+", type=Path)
    create.add_argument("--config", type=Path)
    create.add_argument("--idempotency-key")
    for name in ("advance", "show", "schema"):
        commands.add_parser(name).add_argument("run_id")
    for name in ("annotate", "frontier", "review"):
        child = commands.add_parser(name)
        child.add_argument("run_id")
        child.add_argument("json_file", type=Path)
    release = commands.add_parser("release")
    release.add_argument("run_id")
    release.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    store = RunStore(args.root, args.organization)

    async def provider(units, ontology):
        from app.api import repository
        base = repository.effective_ollama_base_url()
        if not base:
            raise ValueError("Existing Workbench Ollama runtime is not configured")
        return await call_ollama(units, ontology, base, repository.OLLAMA_RETRIEVAL_MODEL, repository.ollama_headers())

    def collector(frontier, output_dir, options):
        return collector_for_runtime(args.collector_python or existing_collector_runtime())(frontier, output_dir, options)
    pipeline = ResearchPipeline(store, provider=provider, collector=collector)
    if args.command == "create":
        result = store.create([{"path": path, "title": path.name} for path in args.documents],
                              validate_config(read_json(args.config) if args.config else default_config()),
                              args.actor, args.idempotency_key)
    elif args.command == "advance":
        result = asyncio.run(pipeline.run(args.run_id))
    elif args.command == "show":
        result = store.describe(args.run_id)
    elif args.command == "schema":
        result = extraction_schema(read_json(store.path(args.run_id) / "inputs.json")["config"]["ontology"])
    elif args.command == "annotate":
        result = pipeline.submit_responses(args.run_id, read_json(args.json_file))
    elif args.command == "frontier":
        result = pipeline.submit_frontier(args.run_id, read_json(args.json_file), args.actor)
    elif args.command == "review":
        result = pipeline.review(args.run_id, read_json(args.json_file), args.actor)
    else:
        result = pipeline.release(args.run_id, args.actor, args.preview)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
