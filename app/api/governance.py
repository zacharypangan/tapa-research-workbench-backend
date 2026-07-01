import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from app.auth import require_permission


router = APIRouter(
    prefix="/governance",
    tags=["governance"],
    dependencies=[Depends(require_permission("progress:read"))],
)

REGISTRY_FIELDS = (
    "dataset_id",
    "dataset_name",
    "title",
    "owner",
    "data_manager",
    "date_received",
    "version",
    "status",
    "access_level",
    "format",
    "system_destination",
    "quality_status",
    "temporal_applicability",
    "temporal_record_count",
    "spatial_applicability",
    "spatial_record_count",
    "transformation_count",
    "resource_count",
    "resource_names",
    "package_path",
    "package_scope",
    "source_basis",
    "quality_notes",
)

RESTRICTED_ACCESS_LEVELS = {"restricted", "permission_required", "embargoed"}


class GovernanceMetadataError(Exception):
    pass


def governance_root() -> Path | None:
    if os.getenv("GOVERNANCE_ENABLED", "").strip().lower() != "true":
        return None

    configured_root = os.getenv("GOVERNANCE_ROOT", "").strip()
    if not configured_root:
        return None

    try:
        root = Path(configured_root).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    return root if root.is_dir() else None


def ensure_within_root(root: Path, path: Path, display_path: str) -> Path:
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise GovernanceMetadataError(
            f"Path escapes the configured governance root: {display_path}"
        ) from exc
    return resolved


def discover_packages(root: Path) -> list[tuple[Path, str, str]]:
    packages: list[tuple[Path, str, str]] = []
    package_roots = (
        (Path("examples"), "example"),
        (Path("data") / "pilots", "local"),
    )

    for relative_root, package_scope in package_roots:
        scan_root = root / relative_root
        if not scan_root.exists():
            continue

        resolved_scan_root = ensure_within_root(
            root,
            scan_root,
            relative_root.as_posix(),
        )
        if not resolved_scan_root.is_dir():
            raise GovernanceMetadataError(
                f"Governance package root is not a directory: {relative_root.as_posix()}"
            )

        try:
            children = sorted(scan_root.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            raise GovernanceMetadataError(
                f"Could not read governance package root: {relative_root.as_posix()}"
            ) from exc

        for child in children:
            display_path = (relative_root / child.name).as_posix()
            package = ensure_within_root(root, child, display_path)
            if not package.is_dir():
                continue

            intake_path = package / "metadata" / "dataset_intake.yaml"
            if intake_path.is_file():
                ensure_within_root(
                    root,
                    intake_path,
                    f"{display_path}/metadata/dataset_intake.yaml",
                )
                packages.append((package, display_path, package_scope))

    return packages


def read_text(
    root: Path,
    path: Path,
    display_path: str,
    *,
    required: bool = True,
) -> str:
    if not path.is_file():
        if required:
            raise GovernanceMetadataError(f"Missing metadata file: {display_path}")
        return ""

    resolved_path = ensure_within_root(root, path, display_path)
    try:
        return resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GovernanceMetadataError(
            f"Could not read metadata file: {display_path}"
        ) from exc


def split_yaml_pair(text: str) -> tuple[str, str]:
    key, raw_value = text.split(":", 1)
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key.strip(), value


def parse_simple_yaml(text: str, display_path: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_list: list[Any] | None = None
    current_item: dict[str, str] | None = None

    for number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if indent == 0 and line.endswith(":") and not line.startswith("- "):
            key = line[:-1].strip()
            data[key] = []
            current_list = data[key]
            current_item = None
            continue

        if line.startswith("- "):
            if current_list is None:
                raise GovernanceMetadataError(
                    f"{display_path}:{number}: list item without parent key"
                )
            item_text = line[2:].strip()
            if ":" in item_text:
                key, value = split_yaml_pair(item_text)
                current_item = {key: value}
                current_list.append(current_item)
            else:
                current_item = None
                current_list.append(item_text)
            continue

        if indent > 0 and current_item is not None and ":" in line:
            key, value = split_yaml_pair(line)
            current_item[key] = value
            continue

        if indent == 0 and ":" in line:
            key, value = split_yaml_pair(line)
            data[key] = value
            current_list = None
            current_item = None
            continue

        raise GovernanceMetadataError(
            f"{display_path}:{number}: unsupported YAML line"
        )

    return data


def read_yaml(root: Path, path: Path, display_path: str) -> dict[str, Any]:
    return parse_simple_yaml(read_text(root, path, display_path), display_path)


def read_json(root: Path, path: Path, display_path: str) -> dict[str, Any]:
    text = read_text(root, path, display_path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GovernanceMetadataError(
            f"Invalid JSON in {display_path} at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(data, dict):
        raise GovernanceMetadataError(
            f"Metadata JSON must contain an object: {display_path}"
        )
    return data


def value(data: dict[str, Any], key: str, default: str = "") -> str:
    return str(data.get(key, default)).strip()


def count_records(data: dict[str, Any]) -> int:
    records = data.get("records", [])
    return len(records) if isinstance(records, list) else 0


def extract_markdown_field(text: str, label: str) -> str:
    pattern = re.compile(
        rf"^\s*-\s*{re.escape(label)}:\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def build_registry_row(
    root: Path,
    package: Path,
    package_path: str,
    package_scope: str,
) -> dict[str, Any]:
    metadata = package / "metadata"
    metadata_path = f"{package_path}/metadata"

    intake = read_yaml(
        root,
        metadata / "dataset_intake.yaml",
        f"{metadata_path}/dataset_intake.yaml",
    )
    temporal = read_yaml(
        root,
        metadata / "temporal_fields.yaml",
        f"{metadata_path}/temporal_fields.yaml",
    )
    spatial = read_yaml(
        root,
        metadata / "spatial_fields.yaml",
        f"{metadata_path}/spatial_fields.yaml",
    )
    datapackage = read_json(
        root,
        metadata / "datapackage.json",
        f"{metadata_path}/datapackage.json",
    )
    quality_text = read_text(
        root,
        metadata / "quality_report.md",
        f"{metadata_path}/quality_report.md",
        required=False,
    )
    transformation_text = read_text(
        root,
        metadata / "transformation_log.md",
        f"{metadata_path}/transformation_log.md",
        required=False,
    )

    resources = datapackage.get("resources", [])
    resource_names = (
        [
            str(resource.get("name", "")).strip()
            for resource in resources
            if isinstance(resource, dict) and str(resource.get("name", "")).strip()
        ]
        if isinstance(resources, list)
        else []
    )

    row = {
        "dataset_id": value(intake, "dataset_id"),
        "dataset_name": value(intake, "dataset_name"),
        "title": value(intake, "title"),
        "owner": value(intake, "owner"),
        "data_manager": value(intake, "data_manager"),
        "date_received": value(intake, "date_received"),
        "version": value(intake, "version"),
        "status": value(intake, "status"),
        "access_level": value(intake, "access_level"),
        "format": value(intake, "format"),
        "system_destination": value(intake, "system_destination"),
        "quality_status": extract_markdown_field(quality_text, "Quality status"),
        "temporal_applicability": value(temporal, "applicability", default="present"),
        "temporal_record_count": count_records(temporal),
        "spatial_applicability": value(spatial, "applicability", default="present"),
        "spatial_record_count": count_records(spatial),
        "transformation_count": len(
            re.findall(r"^##\s+TRANSFORM-", transformation_text, re.MULTILINE)
        ),
        "resource_count": len(resource_names),
        "resource_names": "; ".join(resource_names),
        "package_path": package_path,
        "package_scope": package_scope,
        "source_basis": value(intake, "source_basis"),
        "quality_notes": value(intake, "quality_notes"),
    }
    return {field: row[field] for field in REGISTRY_FIELDS}


def build_registry(root: Path) -> dict[str, Any]:
    datasets = [
        build_registry_row(root, package, package_path, package_scope)
        for package, package_path, package_scope in discover_packages(root)
    ]
    summary = {
        "total_datasets": len(datasets),
        "local_pilots": sum(
            dataset["package_scope"] == "local" for dataset in datasets
        ),
        "in_review": sum(dataset["status"] == "in_review" for dataset in datasets),
        "restricted_flags": sum(
            dataset["access_level"] in RESTRICTED_ACCESS_LEVELS
            for dataset in datasets
        ),
    }
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "generated_at": generated_at,
        "summary": summary,
        "datasets": datasets,
    }


@router.get("/status")
def get_governance_status():
    return {"available": governance_root() is not None}


@router.get("/registry")
def get_governance_registry():
    root = governance_root()
    if root is None:
        raise HTTPException(
            status_code=503,
            detail="Data governance registry is unavailable.",
        )

    try:
        return build_registry(root)
    except GovernanceMetadataError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Governance metadata error: {exc}",
        ) from exc
