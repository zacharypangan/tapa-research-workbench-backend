"""No downloads/GPU: verify the installer rejects unsafe release members."""

import importlib.util
import io
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("pilot_install", Path(__file__).parents[1] / "scripts/minpaku_pilot/install_runtime.py")
INSTALL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALL)


class PilotArchiveTests(unittest.TestCase):
    def test_model_selection_does_not_change_verified_runtime_identity(self):
        pin = json.loads((Path(__file__).parents[1] / "scripts/minpaku_pilot/release.json").read_text())
        self.assertTrue(INSTALL.same_runtime_artifact({**pin, "models": ["previous-model"]}, pin))
        self.assertFalse(INSTALL.same_runtime_artifact({**pin, "archive_sha256": "0" * 64}, pin))

    def archive(self, path, members):
        with tarfile.open(path, "w") as archive:
            for member in members:
                data = b"safe" if member.isfile() else None
                member.size = len(data) if data else 0
                archive.addfile(member, io.BytesIO(data) if data else None)

    def test_installs_files_and_internal_library_link(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = tarfile.TarInfo("bin/ollama")
            binary.mode = 0o755
            library = tarfile.TarInfo("lib/ollama/libtest.so.1")
            link = tarfile.TarInfo("lib/ollama/libtest.so")
            link.type, link.linkname = tarfile.SYMTYPE, "libtest.so.1"
            nested_link = tarfile.TarInfo("lib/ollama/cuda/libtest.so")
            nested_link.type, nested_link.linkname = tarfile.SYMTYPE, "../libtest.so.1"
            self.archive(root / "test.tar", [binary, library, link, nested_link])
            output = root / "output"
            output.mkdir()
            INSTALL.extract_checked(root / "test.tar", output)
            self.assertEqual((output / "bin/ollama").read_bytes(), b"safe")
            self.assertTrue((output / "lib/ollama/libtest.so").is_file())
            self.assertTrue((output / "lib/ollama/cuda/libtest.so").is_file())
            self.assertEqual((output / "bin/ollama").stat().st_mode & 0o777, 0o700)

    def test_rejects_escaping_paths_links_and_special_files_before_writes(self):
        bad = [tarfile.TarInfo("../outside"), tarfile.TarInfo("/tmp/outside")]
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            link = tarfile.TarInfo("lib/ollama/escape")
            link.type, link.linkname = kind, "../../../outside"
            bad.append(link)
        special = tarfile.TarInfo("lib/ollama/device")
        special.type = tarfile.CHRTYPE
        bad.append(special)
        for member in bad:
            with self.subTest(member=member.name, kind=member.type), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.archive(root / "test.tar", [tarfile.TarInfo("bin/ollama"), member])
                output = root / "output"
                output.mkdir()
                with self.assertRaises(ValueError):
                    INSTALL.extract_checked(root / "test.tar", output)
                self.assertEqual(list(output.iterdir()), [])

    def test_rejects_unexpected_roots_and_size_limit(self):
        for name, maximum in (("etc/profile", 100), ("bin/ollama", 1)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.archive(root / "test.tar", [tarfile.TarInfo(name)])
                output = root / "output"
                output.mkdir()
                with self.assertRaises(ValueError):
                    INSTALL.extract_checked(root / "test.tar", output, max_bytes=maximum)


class DocumentEvaluationTests(unittest.TestCase):
    def test_local_evaluation_keeps_source_and_rejects_unsupported_entity(self):
        spec = importlib.util.spec_from_file_location("pilot_evaluate", Path(__file__).parents[1] / "scripts/minpaku_pilot/evaluate_document.py")
        evaluate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluate)
        models = ["qwen3:8b", "qwen3-vl:8b-instruct", "qwen3-embedding:0.6b"]
        candidate = {"entities": [
            {"local_id": "E1", "entity_type": "Concept", "label_original": "石拍",
             "evidence": [{"unit_id": "U1", "quote": "石拍用于制作树皮布。"}]},
            {"local_id": "E2", "entity_type": "Concept", "label_original": "外部知识",
             "evidence": [{"unit_id": "U1", "quote": "石拍用于制作树皮布。"}]}], "assertions": []}

        class Response:
            def __init__(self, data):
                self.data, self.content = data, json.dumps(data).encode()
            def raise_for_status(self):
                pass
            def json(self):
                return self.data

        class Client:
            def __init__(self, **kwargs):
                self.chats = 0
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def get(self, url):
                return Response({"version": "0.40.6"} if url.endswith("version") else
                                {"models": [{"name": model, "digest": "sha256:test"} for model in models]})
            def post(self, url, json):
                if url.endswith("embed"):
                    return Response({"embeddings": [[1.0, 0.0] for _ in json["input"]]})
                self.chats += 1
                data = candidate if self.chats == 1 else {"translation": "Stone beaters are used to make barkcloth."}
                return Response({"done": True, "done_reason": "stop", "message": {"content": evaluate.canonical(data)}})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            text = "石拍用于制作树皮布。"
            unit = {"source_id": "paper", "source_version_id": "paper:v1", "unit_id": "long-registered-unit",
                    "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "locator": {"page": 3}}
            (root / "input.json").write_text(json.dumps({"units": [unit]}))
            argv = ["evaluate_document.py", "--base-url", "http://127.0.0.1:11437", "--input", str(root / "input.json"),
                    "--output", str(root / "output.json")]
            with patch.object(sys, "argv", argv), patch.object(evaluate.httpx, "Client", Client), patch("builtins.print"):
                self.assertEqual(evaluate.main(), 0)
            result = json.loads((root / "output.json").read_text())
            attempt = result["services"]["graph_extraction"]["attempts"][0]
            self.assertEqual((attempt["valid_entities"], attempt["rejected_candidates"]), (1, 1))
            self.assertEqual(attempt["validation"]["entities"][0]["source_version_id"], "paper:v1")
            self.assertEqual(result["services"]["translation"]["source"]["unit_id"], unit["unit_id"])
            self.assertEqual(result["services"]["embeddings"]["batch_size"], 2)
            self.assertFalse(result["services"]["embeddings"]["index_written"])
            self.assertFalse(result["scientific_quality_evaluated"])
            self.assertEqual((root / "output.json").stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
