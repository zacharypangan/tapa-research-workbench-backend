"""Offline snapshot-to-block fidelity audit; never collect or promote claims.

Integrity is checked first using the stdlib collection contract. BeautifulSoup is
loaded only for an audit. Collector extraction/locator helpers are not imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re

from .prepare_graph import ValidationError, validate_collection


NOISE = {"head", "script", "style", "nav", "footer", "form", "noscript", "template"}
BOUNDARIES = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "pre", "blockquote",
              "dt", "dd", "figcaption", "caption", "div", "section", "main", "article", "body",
              "html", "ul", "ol", "dl", "table", "thead", "tbody", "tfoot", "td", "th",
              "figure", "br", "hr"}
POLICY = "dom_blocks_v2_owner_inline_text"


def _excluded(node):
    """Inspect the pristine tree instead of decomposing it and moving locators."""
    for item in [node] + list(node.parents):
        if getattr(item, "name", None) in NOISE or getattr(item, "attrs", {}).get("role") == "navigation":
            return True
    return False


def _text_nodes(root):
    from bs4.element import Comment, Declaration, Doctype, NavigableString, ProcessingInstruction

    return [node for node in root.descendants
            if isinstance(node, NavigableString)
            and not isinstance(node, (Comment, Declaration, Doctype, ProcessingInstruction))
            and str(node).strip() and not _excluded(node)]


def _ordered_text(root):
    """Independently normalize source preorder with explicit boundary closures."""
    from bs4.element import Comment, Declaration, Doctype, NavigableString, ProcessingInstruction, Tag

    pieces, opened = [], [root]
    for node in root.descendants:
        if _excluded(node) or isinstance(node, (Comment, Declaration, Doctype, ProcessingInstruction)):
            continue
        ancestors = []
        for parent in node.parents:
            ancestors.append(parent)
            if parent is root:
                break
        ancestors.reverse()
        common = 0
        while common < min(len(opened), len(ancestors)) and opened[common] is ancestors[common]:
            common += 1
        for old in reversed(opened[common:]):
            if old.name in BOUNDARIES:
                pieces.append(" ")
        opened = ancestors
        if isinstance(node, Tag):
            if node.name in BOUNDARIES:
                pieces.append(" ")
            opened = ancestors + [node]
        elif isinstance(node, NavigableString):
            pieces.append(str(node))
    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def _source_text(node):
    if node.name == "tr":
        cells = [child for child in node.children if getattr(child, "name", None) in {"td", "th"}]
        if cells:
            values = [_ordered_text(cell) for cell in cells]
            return " | ".join(values) if any(values) else ""
    return _ordered_text(node)


def _roots(soup, selector):
    """Reproduce declared scope selection, independently of collector helpers."""
    roots, fallbacks = [], []
    if selector:
        try:
            roots = soup.select(selector)
        except Exception as exc:
            fallbacks.append("invalid_selector:" + type(exc).__name__)
        if not roots:
            fallbacks.append("selector_matched_no_root")
    if not roots:
        roots = [soup.find("main") or soup.find("article") or soup.body or soup]
    selected = {id(node) for node in roots}
    roots = [node for node in roots if not any(id(parent) in selected for parent in node.parents)]
    roots = [node for node in roots if not _excluded(node)]
    if not roots:
        fallbacks.append("selected_root_removed_as_noise")
        roots = [next((node for name in ("main", "article", "body")
                       for node in soup.find_all(name) if not _excluded(node)), soup)]
    return roots, fallbacks


def _inside(node, roots):
    ids = {id(root) for root in roots}
    return id(node) in ids or any(id(parent) in ids for parent in node.parents)


def audit_collection(collection: Path):
    validated = validate_collection(collection)
    import bs4
    import soupsieve

    reports = []
    selector = validated["manifest"]["selectors"]["content"]
    for source in validated["sources"]:
        if source["status"] != "collected":
            continue
        sid = source["source_id"]
        soup = bs4.BeautifulSoup(validated["snapshot_bytes"][sid].decode("utf-8"), "html.parser")
        roots, fallbacks = _roots(soup, selector)
        expected_nodes = [node for root in roots for node in _text_nodes(root)]
        expected = {id(node): index for index, node in enumerate(expected_nodes)}
        owners = {identity: [] for identity in expected}
        blocks, issues, order = [], [], []
        for block in validated["grouped"][sid]:
            locator = block["locator"]
            matches = []
            try:
                matches = [soup] if locator == "[document]" else soup.select(locator)
            except Exception as exc:
                issues.append({"evidence_id": block["evidence_id"], "problem": "invalid_css_locator",
                               "exception": type(exc).__name__})
            result = {"evidence_id": block["evidence_id"], "block_id": block["block_id"],
                      "locator": locator, "locator_kind": "document_root" if locator == "[document]" else "css",
                      "matched_nodes": len(matches), "text_matches": False, "within_selected_scope": False}
            if len(matches) != 1:
                issues.append({"evidence_id": block["evidence_id"], "problem": "locator_does_not_resolve_once"})
            else:
                node = matches[0]
                result["tag_matches"] = node.name == block["tag"]
                result["within_selected_scope"] = _inside(node, roots) and not _excluded(node)
                actual = _source_text(node)
                result["source_normalized_text"] = actual
                result["text_matches"] = actual == block["text"]
                owned = [id(text) for text in _text_nodes(node) if id(text) in expected]
                result["owned_text_nodes"] = len(owned)
                for identity in owned:
                    owners[identity].append(block["evidence_id"])
                    order.append(expected[identity])
                for flag in ("tag_matches", "within_selected_scope", "text_matches"):
                    if not result[flag]:
                        issues.append({"evidence_id": block["evidence_id"], "problem": flag + "_failed"})
            blocks.append(result)
        missing = [identity for identity, owned in owners.items() if not owned]
        overlapping = [identity for identity, owned in owners.items() if len(owned) > 1]
        ordered = order == sorted(order) and len(order) == len(set(order))
        if missing:
            issues.append({"problem": "retained_scope_text_omitted", "text_nodes": len(missing)})
        if overlapping:
            issues.append({"problem": "retained_scope_text_duplicated", "text_nodes": len(overlapping)})
        if not ordered:
            issues.append({"problem": "block_text_ownership_order_mismatch"})
        warnings = source["warnings"]
        reported_policy = source.get("evidence_text_policy") or source.get("text_policy") or source.get("extraction_policy")
        if reported_policy is not None and reported_policy != POLICY:
            issues.append({"problem": "unsupported_reported_text_policy", "reported_policy": reported_policy})
        page_nodes = _text_nodes(soup)
        loading = any(re.search(r"\b(?:loading|please wait)\b", block["text"], re.I)
                      for block in validated["grouped"][sid])
        reports.append({"collector_source_id": sid, "source_url": source["final_url"],
                        "source_version_id": source["source_version_id"], "captured_at": source["captured_at"],
                        "requested_selector": selector, "effective_roots": [
                            {"tag": node.name, "id": node.get("id"), "class": node.get("class", [])} for node in roots],
                        "selection_fallbacks": fallbacks, "collector_warnings": warnings,
                        "reported_text_policy": reported_policy, "software": source.get("software"),
                        "status": "failed" if issues else "passed", "blocks": blocks, "issues": issues,
                        "coverage": {"exported_blocks": len(blocks), "retained_scope_text_nodes": len(expected),
                                     "missing_text_nodes": len(missing), "overlapping_text_nodes": len(overlapping),
                                     "ownership_in_source_order": ordered,
                                     "out_of_scope_retained_text_nodes": len(page_nodes) - len(expected)},
                        "missing_text_examples": [str(expected_nodes[expected[identity]])[:200] for identity in missing[:10]],
                        "overlapping_text_examples": [
                            {"text": str(expected_nodes[expected[identity]])[:200], "owners": owners[identity]}
                            for identity in overlapping[:10]],
                        "content_completeness_diagnostics": {"possible_loading_text": loading,
                            "rendered_dataset_completeness": "not_established", "semantic_recall": "not_measured"}})
    passed = bool(reports) and all(row["status"] == "passed" for row in reports)
    return {"format_version": 1, "audit_kind": "independent_saved_snapshot_dom_fidelity",
            "status": "passed" if passed else "failed" if reports else "no_collected_sources",
            "scope": "Retained DOM text within effective extraction roots, under the declared exclusion/normalization policy.",
            "limits": ["Not page-wide or media coverage; no media is fetched.",
                       "A loading placeholder or table header can be faithful to an incomplete snapshot.",
                       "No rights, source authorship, browser visibility, specialist review or semantic recall is established.",
                       "Original collection and graph eligibility remain unchanged."],
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime": {"python": platform.python_version(), "beautifulsoup": bs4.__version__,
                        "soupsieve": soupsieve.__version__, "parser": "html.parser"},
            "normalization_policy": POLICY, "excluded_elements": sorted(NOISE), "excluded_role": "navigation",
            "input_files": {name: {"sha256": hashlib.sha256(data).hexdigest()}
                            for name, data in sorted(validated["input_bytes"].items())},
            "collection_statuses": validated["manifest"]["statuses"], "collected_source_audits": reports}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.resolve().is_relative_to(args.collection.resolve()):
            raise ValidationError("Audit output must be outside the immutable collection directory")
        if args.output.exists() or args.output.is_symlink():
            raise ValidationError("Choose a new audit output; existing files are not replaced")
        result = audit_collection(args.collection)
        content = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
        args.output.chmod(0o600)
    except (ValidationError, OSError, ImportError) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")
    print(json.dumps({"status": result["status"], "collected_sources": len(result["collected_source_audits"]),
                      "output": str(args.output.absolute())}))
    if result["status"] != "passed":
        parser.exit(1)


if __name__ == "__main__":
    main()
