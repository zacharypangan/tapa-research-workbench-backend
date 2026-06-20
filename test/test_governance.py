import json
import os
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import governance


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(governance.router, prefix="/api/v1")
    return TestClient(app)


def write_package(
    root: Path,
    relative_path: str,
    *,
    dataset_id: str = "DATASET-0001",
    status: str = "in_review",
    access_level: str = "project_internal",
    temporal_records: int = 1,
    spatial_records: int = 1,
) -> Path:
    package = root / relative_path
    metadata = package / "metadata"
    metadata.mkdir(parents=True)

    (metadata / "dataset_intake.yaml").write_text(
        dedent(
            f"""\
            dataset_id: {dataset_id}
            dataset_name: sample_dataset
            title: Sample Dataset
            owner: Dataset Owner
            data_manager: Data Manager
            date_received: 2026-06-15
            version: v01
            status: {status}
            format: CSV
            source_basis: Synthetic fixture
            system_destination: Repository workbench
            quality_notes: Review before reuse
            access_level: {access_level}
            """
        ),
        encoding="utf-8",
    )

    temporal_lines = ["applicability: present", "records:"]
    for index in range(temporal_records):
        temporal_lines.extend(
            [
                f"  - time_id: TIME-{index + 1}",
                "    linked_record_id: ROW-1",
            ]
        )
    (metadata / "temporal_fields.yaml").write_text(
        "\n".join(temporal_lines) + "\n",
        encoding="utf-8",
    )

    spatial_lines = ["applicability: present", "records:"]
    for index in range(spatial_records):
        spatial_lines.extend(
            [
                f"  - place_id: PLACE-{index + 1}",
                "    linked_record_id: ROW-1",
            ]
        )
    (metadata / "spatial_fields.yaml").write_text(
        "\n".join(spatial_lines) + "\n",
        encoding="utf-8",
    )

    (metadata / "datapackage.json").write_text(
        json.dumps(
            {
                "resources": [
                    {"name": "raw_sample"},
                    {"name": "cleaned_sample"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (metadata / "quality_report.md").write_text(
        "- Quality status: provisional\n",
        encoding="utf-8",
    )
    (metadata / "transformation_log.md").write_text(
        "## TRANSFORM-001\n\n## TRANSFORM-002\n",
        encoding="utf-8",
    )
    return package


class GovernanceApiTests(unittest.TestCase):
    def test_status_and_registry_are_unavailable_when_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "GOVERNANCE_ENABLED": "",
                    "GOVERNANCE_ROOT": directory,
                },
                clear=False,
            ):
                client = make_client()
                self.assertEqual(
                    client.get("/api/v1/governance/status").json(),
                    {"available": False},
                )
                response = client.get("/api/v1/governance/registry")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "Data governance registry is unavailable.",
        )

    def test_status_is_unavailable_when_root_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with patch.dict(
                os.environ,
                {
                    "GOVERNANCE_ENABLED": "true",
                    "GOVERNANCE_ROOT": str(missing),
                },
                clear=False,
            ):
                response = make_client().get("/api/v1/governance/status")

        self.assertEqual(response.json(), {"available": False})

    def test_empty_registry_returns_zero_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "GOVERNANCE_ENABLED": "true",
                    "GOVERNANCE_ROOT": directory,
                },
                clear=False,
            ):
                client = make_client()
                status_response = client.get("/api/v1/governance/status")
                registry_response = client.get("/api/v1/governance/registry")

        self.assertEqual(status_response.json(), {"available": True})
        self.assertEqual(registry_response.status_code, 200)
        self.assertEqual(
            registry_response.json()["summary"],
            {
                "total_datasets": 0,
                "local_pilots": 0,
                "in_review": 0,
                "restricted_flags": 0,
            },
        )
        self.assertEqual(registry_response.json()["datasets"], [])

    def test_registry_discovers_supported_packages_and_builds_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "governance"
            write_package(
                root,
                "examples/example_dataset",
                dataset_id="DATASET-EXAMPLE",
                status="active",
                temporal_records=2,
            )
            write_package(
                root,
                "data/pilots/local_dataset",
                dataset_id="DATASET-LOCAL",
                access_level="restricted",
                spatial_records=2,
            )
            ignored = root / "other" / "ignored_dataset" / "metadata"
            ignored.mkdir(parents=True)
            (ignored / "dataset_intake.yaml").write_text(
                "dataset_id: IGNORED\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "GOVERNANCE_ENABLED": "true",
                    "GOVERNANCE_ROOT": str(root),
                },
                clear=False,
            ):
                response = make_client().get("/api/v1/governance/registry")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(
                payload["summary"],
                {
                    "total_datasets": 2,
                    "local_pilots": 1,
                    "in_review": 1,
                    "restricted_flags": 1,
                },
            )
            self.assertEqual(
                [dataset["dataset_id"] for dataset in payload["datasets"]],
                ["DATASET-EXAMPLE", "DATASET-LOCAL"],
            )
            example = payload["datasets"][0]
            self.assertEqual(example["package_path"], "examples/example_dataset")
            self.assertEqual(example["package_scope"], "example")
            self.assertEqual(example["quality_status"], "provisional")
            self.assertEqual(example["temporal_record_count"], 2)
            self.assertEqual(example["resource_count"], 2)
            self.assertEqual(
                example["resource_names"],
                "raw_sample; cleaned_sample",
            )
            self.assertEqual(example["transformation_count"], 2)
            self.assertNotIn(str(root), json.dumps(payload))

    def test_registry_returns_clear_error_for_malformed_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "governance"
            package = write_package(root, "examples/broken")
            (package / "metadata" / "datapackage.json").write_text(
                "{not valid json",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "GOVERNANCE_ENABLED": "true",
                    "GOVERNANCE_ROOT": str(root),
                },
                clear=False,
            ):
                response = make_client().get("/api/v1/governance/registry")

            self.assertEqual(response.status_code, 500)
            detail = response.json()["detail"]
            self.assertIn(
                "Invalid JSON in examples/broken/metadata/datapackage.json",
                detail,
            )
            self.assertNotIn(str(root), detail)

    def test_registry_rejects_package_symlink_outside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            root = temporary_root / "governance"
            (root / "examples").mkdir(parents=True)
            outside = temporary_root / "outside"
            write_package(outside, "escaped")
            (root / "examples" / "escaped").symlink_to(outside / "escaped")
            with patch.dict(
                os.environ,
                {
                    "GOVERNANCE_ENABLED": "true",
                    "GOVERNANCE_ROOT": str(root),
                },
                clear=False,
            ):
                response = make_client().get("/api/v1/governance/registry")

            self.assertEqual(response.status_code, 500)
            detail = response.json()["detail"]
            self.assertIn(
                "Path escapes the configured governance root: examples/escaped",
                detail,
            )
            self.assertNotIn(str(temporary_root), detail)


if __name__ == "__main__":
    unittest.main()
