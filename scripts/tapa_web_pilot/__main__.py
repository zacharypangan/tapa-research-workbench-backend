import argparse
import json
from pathlib import Path

from .core import Config, collect


def main():
    parser = argparse.ArgumentParser(description="Collect source-linked web evidence for tapa research.")
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engine", choices=["static", "selenium", "playwright", "auto"], default="static")
    parser.add_argument("--extractor", choices=["bs4", "trafilatura"], default="bs4")
    parser.add_argument("--allowed-host", action="append", default=[])
    parser.add_argument("--content-selector")
    parser.add_argument("--wait-selector", default="body")
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--browser-binary")
    parser.add_argument("--driver-path")
    args = parser.parse_args()
    if not 1 <= args.max_pages <= 100 or not 0 <= args.max_depth <= 2:
        parser.error("max-pages must be 1-100 and max-depth 0-2")
    if not 0 <= args.delay <= 60 or not 1 <= args.timeout <= 120:
        parser.error("delay must be 0-60 and timeout 1-120 seconds")
    config = Config(**vars(args))
    try:
        summary = collect(config)
    except (ValueError, ImportError, OSError) as exc:
        parser.exit(2, f"Pilot configuration failed: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["successful_sources"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
