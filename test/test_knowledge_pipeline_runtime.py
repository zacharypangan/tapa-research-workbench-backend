"""Runtime bridge tests use mocks or dependency checks, never live scraping."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.repository.knowledge_pipeline import runtime


FRONTIER = [{"url": "https://example.org/item", "seed_ids": ["P1:entity"], "question": "Reported material?"}]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def fake_worker(self, payload=None, returncode=0):
        calls = []

        def run(arguments, **kwargs):
            calls.append((arguments, kwargs))
            result_path = Path(arguments[arguments.index("--result") + 1])
            runtime._write_json(result_path, payload if payload is not None else {
                "ok": True, "result": {"sources": [], "units": [], "attempts": []}, "python_version": "fixture"})
            return SimpleNamespace(returncode=returncode, stdout="PRIVATE_STDOUT", stderr="PRIVATE_STDERR")

        return run, calls

    def test_discovery_prefers_current_then_existing_approved_runtime(self):
        approved = self.base / "approved-runtime" / "bin" / "python"
        with patch.object(runtime, "APPROVED_LOCAL_RUNTIME", approved), patch.object(runtime, "_available_runtime", side_effect=[False, True]) as check:
            selected = runtime.existing_collector_runtime()
        self.assertEqual(selected, approved.absolute())
        self.assertEqual(check.call_args_list[0].args, (Path(sys.executable),))
        self.assertEqual(check.call_args_list[1].args, (approved,))
        with patch.object(runtime, "_available_runtime", return_value=True) as check:
            self.assertEqual(runtime.existing_collector_runtime(), Path(sys.executable).absolute())
        check.assert_called_once_with(Path(sys.executable))
        with patch.object(runtime, "_available_runtime", return_value=False):
            with self.assertRaisesRegex(runtime.CollectorRuntimeError, "No installed collector runtime"):
                runtime.existing_collector_runtime()

    def test_discovery_probe_is_bounded_and_never_discloses_output(self):
        with patch.object(runtime.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="SECRET", stderr="SECRET")) as run:
            self.assertFalse(runtime._available_runtime(Path(sys.executable)))
        args, kwargs = run.call_args
        self.assertEqual(args[0][1:], ["-c", "import bs4, httpx"])
        self.assertEqual(kwargs["timeout"], 10)
        self.assertTrue(kwargs["capture_output"])

    def test_nonexistent_interpreter_is_explicit(self):
        with self.assertRaisesRegex(runtime.CollectorRuntimeError, "existing executable"):
            runtime.collector_for_runtime(self.base / "missing-python")

    def test_preserves_venv_symlink_path(self):
        link = self.base / "venv-python"
        link.symlink_to(Path(sys.executable))
        self.assertEqual(runtime._python_path(link), link.absolute())

    def test_collector_request_contract_environment_hashes_and_fresh_target(self):
        worker, calls = self.fake_worker()
        output = self.base / "captures" / "round-001"
        with patch.object(runtime.subprocess, "run", side_effect=worker), patch.dict(runtime.os.environ, {"PYTHONPATH": "retained-path"}):
            result = runtime.collector_for_runtime(Path(sys.executable))(FRONTIER, output, {"max_pages": 1})
        args, kwargs = calls[0]
        self.assertEqual(args[1:4], ["-m", "app.repository.knowledge_pipeline.runtime", "collect"])
        self.assertEqual(kwargs["env"]["PYTHONPATH"], str(runtime.BACKEND_ROOT) + runtime.os.pathsep + "retained-path")
        self.assertEqual(kwargs["cwd"], runtime.BACKEND_ROOT)
        self.assertTrue(kwargs["capture_output"])
        self.assertLessEqual(kwargs["timeout"], runtime.MAX_COLLECTION_SECONDS)
        execution = result["runtime_execution"]
        request = Path(execution["request_path"])
        self.assertEqual(request.parent, output.parent)
        self.assertEqual(json.loads(request.read_text()), {"frontier": FRONTIER, "options": {"max_pages": 1}, "output_dir": str(output)})
        self.assertEqual(request.stat().st_mode & 0o222, 0)
        self.assertTrue(Path(execution["result_path"]).is_file())
        self.assertEqual(len(execution["request_sha256"]), 64)
        self.assertNotIn("PRIVATE", str(result))
        self.assertFalse(output.exists(), "Bridge must leave destination creation to the existing collector")

    def test_collection_timeout_and_api_executable_options_are_bounded(self):
        front = [{**FRONTIER[0], "url": f"https://example.org/{index}"} for index in range(25)]
        self.assertEqual(runtime._collection_timeout(front, {"max_pages": 25, "timeout": 120, "delay": 60}), 1800)
        self.assertEqual(runtime._collection_timeout(FRONTIER, {}), 91)
        for options in ({"max_pages": 26}, {"max_pages": 0}, {"timeout": float("inf")}, {"collector_python": "/path"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                runtime._collection_timeout(FRONTIER, options)
        with self.assertRaisesRegex(ValueError, "explicit nonempty seed_ids"):
            runtime._collection_timeout([{"url": "https://example.org/item", "question": "Why?"}], {})

    def test_existing_collection_cannot_be_overwritten(self):
        output = self.base / "collection"
        output.mkdir()
        with patch.object(runtime.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "fresh collection"):
                runtime.collector_for_runtime(Path(sys.executable))(FRONTIER, output, {})
        run.assert_not_called()

    def test_timeout_preserves_request_without_disclosing_subprocess_output(self):
        expired = subprocess.TimeoutExpired(["python"], 91, output="PRIVATE_STDOUT", stderr="PRIVATE_STDERR")
        with patch.object(runtime.subprocess, "run", side_effect=expired):
            with self.assertRaisesRegex(runtime.CollectorRuntimeError, "bounded 91-second") as raised:
                runtime.collector_for_runtime(Path(sys.executable))(FRONTIER, self.base / "collection", {})
        self.assertNotIn("PRIVATE", str(raised.exception))
        self.assertEqual(len(list(self.base.glob("runtime-request-*.json"))), 1)

    def test_missing_dependency_and_validation_errors_remain_typed(self):
        for kind, error_type in (("ImportError", ImportError), ("ValueError", ValueError), ("RuntimeError", runtime.CollectorRuntimeError)):
            worker, _ = self.fake_worker({"ok": False, "error": {"type": kind, "message": "PRIVATE"}}, returncode=2)
            with self.subTest(kind=kind), patch.object(runtime.subprocess, "run", side_effect=worker):
                with self.assertRaises(error_type) as raised:
                    runtime.collector_for_runtime(Path(sys.executable))(FRONTIER, self.base / kind, {})
                self.assertNotIn("PRIVATE", str(raised.exception))

    def test_missing_result_is_not_replaced_with_stdout(self):
        with patch.object(runtime.subprocess, "run", return_value=SimpleNamespace(returncode=2, stdout="SECRET", stderr="SECRET")):
            with self.assertRaisesRegex(runtime.CollectorRuntimeError, "without a structured result") as raised:
                runtime.collector_for_runtime(Path(sys.executable))(FRONTIER, self.base / "capture", {})
        self.assertNotIn("SECRET", str(raised.exception))

    def test_worker_cli_serializes_adapter_result_and_failure(self):
        request, result = self.base / "request.json", self.base / "result.json"
        runtime._write_json(request, {"frontier": FRONTIER, "options": {}, "output_dir": str(self.base / "capture")})
        expected = {"sources": [], "units": [], "attempts": []}
        with patch("app.repository.knowledge_pipeline.adapters.collect_frontier", return_value=expected) as collect:
            code = runtime.main(["collect", "--request", str(request), "--result", str(result)])
        self.assertEqual(code, 0)
        collect.assert_called_once_with(FRONTIER, self.base / "capture", {})
        self.assertEqual(json.loads(result.read_text())["result"], expected)
        self.assertEqual(runtime.main(["collect", "--request", str(request), "--result", str(result)]), 2)
        failure = self.base / "failure.json"
        with patch("app.repository.knowledge_pipeline.adapters.collect_frontier", side_effect=ImportError("PRIVATE dependency detail")):
            code = runtime.main(["collect", "--request", str(request), "--result", str(failure)])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(failure.read_text())["error"], {"type": "ImportError"})
        self.assertNotIn("PRIVATE", failure.read_text())

    def test_real_subprocess_can_import_worker_without_network(self):
        python = runtime.APPROVED_LOCAL_RUNTIME
        if not python.is_file():
            self.skipTest("Existing approved collector runtime is unavailable")
        # Adapter rejects depth before any runtime import or network access.
        with self.assertRaises(ValueError):
            runtime.collector_for_runtime(python)(FRONTIER, self.base / "capture", {"max_depth": 1})
        result = json.loads(next(self.base.glob("runtime-result-*.json")).read_text())
        self.assertEqual(result["error"], {"type": "ValueError"})
        self.assertFalse((self.base / "capture").exists())


if __name__ == "__main__":
    unittest.main()
